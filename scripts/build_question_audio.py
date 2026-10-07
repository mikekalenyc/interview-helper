#!/usr/bin/env python3
"""Build a local, offline interview practice page using existing Kokoro weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


def _prose(text: str) -> str:
    # Remove only boundary separators; retain Markdown and internal whitespace.
    lines = text.strip().splitlines()
    while lines and (not lines[-1].strip() or lines[-1].strip() == "---"):
        lines.pop()
    return "\n".join(lines).strip()


def read_questions(source: Path) -> list[dict[str, object]]:
    text = source.read_text(encoding="utf-8")
    questions: list[dict[str, object]] = []
    section = "Interview questions"
    if re.search(r"^####\s+Q", text, re.MULTILINE):
        headings = list(re.finditer(r"^(#{2,4}) ([^\n]+)$", text, re.MULTILINE))
        local_number = 0
        sections: set[str] = set()
        for index, heading in enumerate(headings):
            level, title = heading.groups()
            if level == "##":
                section = title.strip()
                local_number = 0
                continue
            if level != "####":
                continue
            match = re.fullmatch(r"Q(\d+)\. (.+)", title)
            if not match or int(match[1]) != local_number + 1:
                raise ValueError(f"Malformed, missing or duplicate question in {section}")
            if local_number == 0:
                if section in sections:
                    raise ValueError(f"Duplicate question section: {section}")
                sections.add(section)
            local_number = int(match[1])
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            body = text[heading.end():end]
            marker = re.compile(r"^\*\*Follow-up (question|answer):\*\*[ \t]*", re.MULTILINE)
            markers = list(marker.finditer(body))
            marker_lines = re.findall(r"^\*\*Follow-up\b.*$", body, re.MULTILINE)
            if len(marker_lines) != 2 or [m[1] for m in markers] != ["question", "answer"]:
                raise ValueError(f"Missing or malformed follow-up in {section} Q{local_number}")
            first, second = markers
            answer = _prose(body[:first.start()])
            followup = _prose(body[first.end():second.start()])
            followup_answer = _prose(body[second.end():])
            if not all((answer, followup, followup_answer)):
                raise ValueError(f"Empty answer or follow-up in {section} Q{local_number}")
            questions.append({"number": len(questions) + 1, "section": section,
                              "local_number": local_number, "text": match[2],
                              "answer": answer, "followup_text": followup,
                              "followup_answer": followup_answer})
        if not questions:
            raise ValueError("No question groups found")
        return questions
    if re.search(r"^\d+\.\s+\*\*.+\*\*\s*$", text, re.MULTILINE) and "**Answer:**" in text:
        return read_bold_questions(text)
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block.startswith("## "):
            section = block[3:].strip()
        match = re.fullmatch(r"(\d+)\.\s+(.+)", block, re.DOTALL)
        if match:
            questions.append({"number": int(match[1]), "section": section,
                              "text": re.sub(r"\s+", " ", match[2])})
    if [q["number"] for q in questions] != list(range(1, 51)):
        raise ValueError("Expected exactly 50 questions numbered 1 through 50")
    return questions


def read_bold_questions(text: str) -> list[dict[str, object]]:
    """Numbered bold questions, each followed by a "**Answer:**" paragraph."""
    questions: list[dict[str, object]] = []
    section = "Interview questions"
    question: dict[str, object] | None = None
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        heading = re.fullmatch(r"##\s+(?:\d+\.\s+)?(.+)", block)
        if heading:
            section = heading[1].strip()
            continue
        match = re.fullmatch(r"(\d+)\.\s+\*\*(.+)\*\*", block, re.DOTALL)
        if match:
            question = {"number": int(match[1]), "section": section,
                        "text": re.sub(r"\s+", " ", match[2]).strip()}
            questions.append(question)
            continue
        if block.startswith("**Answer:**"):
            if question is None or "answer" in question:
                raise ValueError("Answer without a preceding question")
            question["answer"] = block[len("**Answer:**"):].strip()
    if [q["number"] for q in questions] != list(range(1, len(questions) + 1)) or not questions:
        raise ValueError("Questions must be numbered consecutively from 1")
    missing = [q["number"] for q in questions if not q.get("answer")]
    if missing:
        raise ValueError(f"Questions without answers: {missing}")
    return questions


def audio_signature(speech: str, model_hash: str, voice_hash: str, config_hash: str) -> str:
    return hashlib.sha256(json.dumps([speech, model_hash, voice_hash, config_hash,
                                     "speed=1.0"], ensure_ascii=False).encode()).hexdigest()


def cache_matches(prior: dict[str, object], prefix: str, signature: str, destination: Path) -> bool:
    return prior.get(prefix + "signature") == signature and destination.is_file()


def spoken_text(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}/\d+\b",
                  lambda m: m[0].replace(".", " dot ").replace("/", " slash "), text)
    terms = {"AWS": "A W S", "VPCs": "V P Cs", "VPC": "V P C",
             "TGW": "T G W", "ZPA": "Z P A", "CIDRs": "ciders", "CIDR": "cider",
             "GCP": "G C P", "ASNs": "A S Ns", "ASN": "A S N", "DR": "disaster recovery",
             "BGP": "B G P", "DNS": "D N S", "IP": "I P",
             "VPN": "V P N", "NACLs": "network access control lists",
             "gNMI": "G N M I", "SSH/CLI": "S S H, C L I",
             "SNMP": "S N M P", "NETCONF": "net conf", "APIs": "A P Is",
             "ON_CHANGE": "on change", "SAMPLE": "sample",
             "L1": "level one", "L2": "level two", "L3": "level three",
             "MX": "M X", "SD-WAN": "S D wan", "WAN": "wan",
             "OOB": "out of band", "NOC": "knock", "PoC": "proof of concept",
             "NAT": "nat", "HA": "high availability", "RFC1918": "R F C nineteen eighteen",
             "ASA/FTD": "A S A or F T D", "ASA/LINA": "A S A Lina", "LINA": "Lina",
             "FTD": "F T D", "FMC": "F M C", "ASA": "A S A", "ACLs": "A C Ls", "ACL": "A C L",
             "ACE": "A C E", "ASP": "A S P", "asp": "A S P", "xlate": "ex late",
             "packet-tracer": "packet tracer", "ICMP": "I C M P", "ISP": "I S P",
             "MSS": "M S S", "MTU": "M T U", "NTP": "N T P", "PAT": "pat",
             "SYN-ACK": "sin ack", "SYN": "sin", "ACK": "ack", "TCP": "T C P", "UDP": "U D P"}
    pattern = r"(?<!\w)(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + r")(?!\w)"
    return re.sub(pattern, lambda m: terms[m[0]], text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Interview practice · Questions and follow-ups")
    parser.add_argument("--eyebrow", default="Your interview practice room")
    parser.add_argument("--heading", default="Practice the answer.<br>Be ready for the follow-up.")
    parser.add_argument("--intro", default=(
        "Play a question, practice your response, and review the suggested answer from your guide. "
        "Then play its follow-up when you’re ready. Each recording stops at the end so you have time to answer."))
    parser.add_argument("--hide-answers", action="store_true",
                        help="start with suggested answers hidden")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    questions = read_questions(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audio").mkdir(exist_ok=True)
    manifest_path = args.output / "questions.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cached = {q["number"]: q for q in old.get("questions", [])}
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(args.model / "config.json"),
                   model=str(args.model / "kokoro-v1_0.pth")).to("cpu").eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model, device="cpu")
    voice = args.model / "voices/af_heart.pt"
    model_hash = hashlib.sha256((args.model / "kokoro-v1_0.pth").read_bytes()).hexdigest()
    voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
    config_hash = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    manifest = {"model": "Kokoro-82M", "voice": "af_heart", "speed": 1.0,
                "source": str(args.source),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "model_sha256": model_hash, "voice_sha256": voice_hash,
                "config_sha256": config_hash, "questions": questions}
    for number, question in enumerate(questions, 1):
        prior = cached.get(question["number"], {})
        for prefix, field in [("", "text"), ("followup_", "followup_text")]:
            if field not in question:
                continue
            speech = spoken_text(str(question[field]))
            suffix = "-followup" if prefix else ""
            relative = f"audio/question-{number:02d}{suffix}.wav"
            destination = args.output / relative
            signature = audio_signature(speech, model_hash, voice_hash, config_hash)
            started = time.perf_counter()
            if not cache_matches(prior, prefix, signature, destination):
                with torch.inference_mode():
                    chunks = [audio.detach().cpu().numpy().reshape(-1)
                              for _, _, audio in pipeline(speech, voice=str(voice), speed=1.0, split_pattern=r"\n+")
                              if audio is not None]
                if not chunks:
                    raise RuntimeError(f"No audio for question {number}{suffix}")
                audio = np.concatenate(chunks)
                if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
                    raise RuntimeError(f"Invalid audio for question {number}{suffix}")
                temporary = destination.with_suffix(".tmp.wav")
                sf.write(temporary, audio, 24000, subtype="PCM_16")
                temporary.replace(destination)
            info = sf.info(destination)
            question.update({prefix + "audio": relative, prefix + "duration": round(info.duration, 2),
                             prefix + "spoken_text": speech, prefix + "signature": signature})
            print(f"Question {number:02d}{suffix}: {info.duration:.1f}s audio; {time.perf_counter()-started:.1f}s build", flush=True)
        temporary_manifest = manifest_path.with_suffix(".tmp.json")
        temporary_manifest.write_text(json.dumps(manifest, indent=2))
        temporary_manifest.replace(manifest_path)
    template = Path(__file__).with_name("question_player.html").read_text()
    page = {"__PAGE_TITLE__": args.title, "__EYEBROW__": args.eyebrow, "__HEADING__": args.heading,
            "__INTRO__": args.intro, "__ANSWERS_CHECKED__": "" if args.hide_answers else " checked",
            "__LIST_CLASS__": " answers-hidden" if args.hide_answers else "",
            "__QUESTIONS_JSON__": json.dumps(questions).replace("<", "\\u003c")}
    for placeholder, value in page.items():
        template = template.replace(placeholder, value)
    (args.output / "index.html").write_text(template)
    print(f"Ready: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
