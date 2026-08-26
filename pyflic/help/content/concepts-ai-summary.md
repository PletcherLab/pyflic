# AI summary

An optional, AI-written narrative of a Project's Combined Analysis, up to about a page.

## What it is given

Exactly what the report's reader sees: the cover text, the pooled figures as images, the
statistics table, and the per-chamber summary CSVs. Never the rendered PDF, and never raw
per-sample signal data.

## What it must not do

It **summarizes** the pipeline's analysis; it never performs its own. The instructions tell
it that every number it states must appear in the input, that it must say plainly when a
comparison is not significant, and that it should point out where the pooled and
mixed-model p-values disagree — that gap is between-member variation, and it matters.

## It is a derivative of one run

An AI summary describes one Combined Analysis's numbers. Rebuilding the Combined Analysis
**deletes** it, so a stale narrative can never sit beside fresh numbers. Generate it again
afterwards if you still want one.

## Providers and keys

Anthropic or OpenAI, chosen in the AI panel. A provider is offered only when its API key is
present:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

in your environment or a `.env` file, either beside your project or in
`~/.config/pyflic/.env`. Real environment variables always win.

Presence of a key is all that is checked — validity is only discovered when you actually
ask for a summary. A failed generation never blocks a report; you get an error message
instead.
