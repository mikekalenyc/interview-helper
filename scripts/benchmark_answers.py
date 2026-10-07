"""Measure text-to-complete-answer latency; output contains private evidence.

Run with the project installed. Does not capture audio or establish hardware latency.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import httpx

from interview_helper.answer import (AnswerTurn, build_grounded_messages, needs_technical_reference,
                                     CANDIDATE_BUDGET, REFERENCE_BUDGET, REFERENCE_LIMIT, ANSWER_TOKEN_LIMIT)
from interview_helper.context import CandidateProfile
from interview_helper.library import TechnicalLibrary
from interview_helper.qwen import DEFAULT_BASE_URL, DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--context', type=Path, action='append', default=[])
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True, help='JSON array of question strings')
    parser.add_argument('--output', type=Path, required=True, help='New private JSONL file; never overwritten')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    if not isinstance(questions, list) or not all(isinstance(q, str) and q.strip() for q in questions):
        parser.error('Questions must be a JSON array of nonempty strings')
    profile = CandidateProfile.load(args.resume, tuple(args.context))
    library = TechnicalLibrary.open(args.library, cache_path=args.cache)
    history: list[AnswerTurn] = []
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as output, httpx.Client(timeout=45) as http:
            for question in questions:
                started = time.perf_counter()
                recent = history[-1].question if history else ''
                chunks = profile.fast_chunks(question, recent_question=recent, max_characters=CANDIDATE_BUDGET)
                refs = library.search(question, recent_question=recent, max_characters=REFERENCE_BUDGET, limit=REFERENCE_LIMIT) if needs_technical_reference(question, chunks) else ()
                messages = build_grounded_messages(question, profile, history, guide_chunks=chunks, references=refs)
                retrieval = time.perf_counter() - started
                response = http.post(args.base_url.rstrip('/') + '/chat/completions', json={
                    'model': args.model, 'messages': messages, 'stream': False,
                    'max_tokens': ANSWER_TOKEN_LIMIT, 'chat_template_kwargs': {'enable_thinking': False},
                })
                response.raise_for_status()
                body = response.json()
                elapsed = time.perf_counter() - started
                choice = body['choices'][0]
                answer = choice['message'].get('content')
                completed = choice.get('finish_reason') == 'stop' and isinstance(answer, str) and bool(answer.strip())
                result = dict(question=question, answer=answer, completed=completed,
                              total_seconds=elapsed, retrieval_seconds=retrieval,
                              finish_reason=choice.get('finish_reason'), usage=body.get('usage'),
                              timings=body.get('timings'), evidence=[c.text for c in chunks],
                              references=[dict(text=r.text, url=r.url, locator=r.locator) for r in refs])
                output.write(json.dumps(result) + '\n')
                output.flush()
                print(f'{elapsed:.3f}s complete={completed}', flush=True)
                if completed:
                    history.append(AnswerTurn(question, answer))
    finally:
        library.close()


if __name__ == '__main__':
    main()
