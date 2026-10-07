# Conversational and technical modes

The desktop has two answer modes. **Conversational mode** is the normal/general
mode, with **Enable technical mode** unchecked. Check that box for **Technical
mode**. Change modes in Setup between interviews; the choice is locked during
an active session.

Both modes use the same audio workflow: headphone output supplies interviewer
questions, your microphone supplies what you actually say, and **Start Interview**
/ **Interview Done** control both inputs. Neither mode requires a hotkey. Both
can use the local or OpenAI answer provider and require a résumé to generate
suggestions. A mode changes how answers are prepared, not who is recorded.

## Compare the functions

| Function | Conversational mode | Technical mode |
| --- | --- | --- |
| Best fit | Background, behavioral questions, project walkthroughs, mixed interviews | Focused technical definitions, comparisons, troubleshooting, and procedures |
| Primary preparation | Factual résumé, project notes, and personal Q&A | Prepared technical Q&A plus résumé/background for personal questions |
| How Q&A is used | Relevant excerpts are retrieved with surrounding topic context | The whole selected Q&A file is supplied, up to 40,000 characters |
| Source of personal claims | Résumé, background, and actual spoken answers | Résumé, background, and actual spoken answers; technical Q&A is not proof of experience |
| Uncovered technical questions | Uses relevant supplied context and optional local reference excerpts; prompts acknowledge unsupported gaps | May use the model's technical knowledge and correct inaccurate prepared answers |
| Typical answer shape | Direct short answers; project walkthroughs add role, business reason, actions, and supported outcomes | One or two short sentences, about 15–30 words, with at most three key points; up to four sentences for steps |
| Follow-ups | Bounded recent suggestions and selected actual speech, including the latest answer | Up to the last five answered questions paired with actual speech, within a 12,000-character budget |
| Technical library | Optional lookup when a compatible local library is selected | Desktop library lookup is paused; its folder is remembered for switching back |
| Preferred wording | Source answers provide facts to condense; fixed plain-speaking rules shape the result | Matching prepared answers guide useful points and plain wording, subject to correctness and length |

These are prompt targets and context limits, not guarantees of accuracy or exact
wording. Technical mode does not read answers verbatim. Conversation context is
bounded; neither mode remembers every prior turn indefinitely.

## Prepare questions for conversational mode

Prepare **answers about your real experience**, not just a list of questions.
Use [`conversational-answers.example.md`](../examples/conversational-answers.example.md)
and keep your résumé and project background in separate files if that is clearer.

1. Write a factual résumé with your roles, responsibilities, and skills.
2. For each important project, explain the situation, business reason, your role,
   actions, constraints, and observed outcome. Preserve what other people owned.
3. Add likely questions beneath topic headings, immediately followed by your
   answers in the language you would naturally speak.
4. Add likely follow-ups: why you chose an approach, what went wrong, what you
   personally changed, and what the result was. Mark unknown details honestly.
5. In Setup, leave **Enable technical mode** unchecked, choose **Résumé**, and add
   the background and conversational Q&A under **Interview guides and background**.

Start with around 10–20 representative Q&A pairs as a preparation suggestion;
there is no required question count. Include an introduction, a project
walkthrough, a difficult problem, teamwork, a tradeoff, and an unknown-experience
question. Topic-specific Markdown headings help retrieval.

Template structure:

```markdown
# My interview background

## Project: [your project]

### Background
[Why the project happened, your role, and what you actually worked on.]

### Question: Walk me through your part in the project.
[Your role first, then the reason for the work and your actual actions.]

### Follow-up: Why did you choose that approach?
[Your real reasoning and constraints.]

### Follow-up: What was the result?
[Observed outcome. If no measurement exists, say that instead of adding one.]
```

The software does not require the literal words `Question` or `Follow-up`; a
heading containing the question and an answer immediately below is sufficient.
Keep each answer with the topic it describes so retrieval can preserve context.

All supplied personal Q&A is treated as factual background. Calling a section
“practice” does not exclude it from evidence. Replace placeholders and remove
fictional stories before loading a file. If an example is hypothetical, state
that clearly and avoid wording it as something you actually did. Do not mix in
someone else's story merely because you like their speaking style.

Example launch, with provider selection handled separately in Setup:

```bash
.venv/bin/interview-helper-gui \
  --resume "$HOME/private/interview/resume.md" \
  --context "$HOME/private/interview/background.md" \
  --context "$HOME/private/interview/conversational-answers.md"
```

## Prepare questions for technical mode

Prepare **accurate technical answers in your preferred concise wording**. Start
from [`technical-answers.example.md`](../examples/technical-answers.example.md).

1. Choose the role's subjects: for example, routing, firewalls, DNS, or automation.
2. Group definitions, comparisons, troubleshooting, and configuration questions
   under descriptive topic headings.
3. Write one or two short sentences per ordinary answer, with only the most useful
   one to three points. Keep standard technical terms; use everyday connecting
   words. For a procedure, a short sequence of up to four sentences is reasonable.
4. Include likely follow-ups as separate Q&A entries. Name the technology in each
   question so “Why?” or “How does that work?” has a clear topic in the file.
5. Verify the answers yourself. Note product/version qualifications where needed.
   Keep the entire selected file at or below 40,000 characters.
6. In Setup, select a résumé, check **Enable technical mode**, and choose the file
   under **Technical answers**. Continue supplying genuine personal background
   separately if you want experience questions answered accurately.

Example prepared content:

```markdown
# Technical interview answers

## Networking basics

### What does ARP do?
ARP maps an IPv4 address to a MAC address on the local network.

### How does a host learn a MAC address with ARP?
It broadcasts an ARP request. The host using that IPv4 address replies with its MAC.

### How do TCP and UDP differ?
TCP provides an ordered byte stream with retransmission. UDP sends datagrams
without built-in delivery or ordering guarantees.
```

No exact numbering scheme or keyword matching is required. The model receives
the full prepared file and decides which answer applies. A question-only file
provides no preferred answer wording; write the answers as well. A plain text
file is accepted, but Markdown is easier to organize and review.

Example launch:

```bash
.venv/bin/interview-helper-gui \
  --resume "$HOME/private/interview/resume.md" \
  --context "$HOME/private/interview/background.md" \
  --technical-answers "$HOME/private/interview/technical-answers.md"
```

`--technical-answers` preselects the checkbox. Do not also add this technical Q&A
as personal context simply to make it available; the technical field already
supplies it. Unchecking technical mode keeps the selected Q&A path for reuse and
restores the general library setting. File selection is not saved across app
launches unless your launcher supplies the arguments.

## Follow-ups and wording in either mode

Your actual spoken answer is separate from the generated suggestion. If you
answer differently, the microphone transcript supplies that difference to later
questions. Review transcription in **You said** and use correction/discard when
needed. A correction affects the next question; it does not automatically request
another answer. **History** saves completed text for review and is not loaded
into a new interview's live context.

Preferred Q&A wording helps, particularly in technical mode, but a dedicated
style profile is a proposed feature. A worksheet for that future feature is
provided separately in [personalization](PERSONALIZATION.md). Current prompts
still control length and plain spoken style, and may rewrite prepared answers.

After updating preparation files, end and restart the session. Check the setup
with a prepared question, a paraphrase, a follow-up, an unrelated new topic, and
a question whose personal answer is unknown. Review facts and natural wording
before relying on the suggestions.
