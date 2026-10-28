# AI-assisted Nursing Charting PoC

**Experimental, synthetic-data-only demonstration. Not for clinical use.**

A local Python proof of concept for converting short natural-language nursing observations into draft nursing notes, with explicit human review and approval before an example can be saved as a reusable reference (Gold).

## Design

1. **Exact** — reuse a reference record only for identical input and context; hold ambiguous conflicts.
2. **Learned** — try conservatively discovered/approved numeric templates where supported. This small public demo may have no active patterns.
3. **LLM fallback** — when deterministic routes cannot handle an input, call an external model for a draft, with relevant synthetic examples as style references.
4. **Human review** — generated drafts are editable; explicit approval is required to add them to the local Gold store.

Neither automated output screening nor a Gold label establishes clinical correctness. This is **not** connected to an EMR and has not demonstrated clinical accuracy or workflow time savings.

## Public demonstration dataset

`examples/public-demo-gold.json` contains **3 newly written synthetic illustrative examples**, not the private project training set, and not clinician-validated clinical ground truth. The original private data and classification audit files are **not distributed**. The small demo does not reproduce the original evaluation results or complete classification behavior.

## Run locally (Windows)

- Python **3.12+** is required. No extra third-party packages are required.
- Double-click `start_windows.cmd`. On first launch, supply **your own** OpenAI API key in the terminal. It is saved to a local `.env` file and is never intended for Git.
- If the local store does not exist, the launcher restores the **3 synthetic demo examples** into `.nursing-data/gold.jsonl` (not tracked by Git).
- The local UI opens at `http://127.0.0.1:8765`. The default model for the LLM route is `gpt-6-luna`; external calls can incur charges and require model access.
- To demonstrate Exact without an API request, try the input: `체온 36.8도 측정함. 통증 없다고 말함.`
- To stop, press Ctrl+C in the server terminal.

Do not enter actual patient data. The app is deliberately bound to the local loopback interface, not deployed as a hosted web service. An existing `.nursing-data/gold.jsonl` file is preserved rather than overwritten.

## Verification and limitations

The original private PoC explored conservative exact matching, numeric pattern discovery, conflict handling, structural output checks, and human approval. Public example count is intentionally too small to measure generalization or clinical reliability. The UI and underlying implementation are provided for code review; safety and effectiveness are not established.

## Public source boundary

No original Gold corpus, original classification manifests, API keys, local drafts, or hospital data are included. This repository was created separately from the private development repository, without copying its Git history.
