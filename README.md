# NEXUS

**A local-first AI agent for Windows.** Runs a 3B language model entirely on your own machine — no cloud, no API key, no data leaving your laptop. Opens as a compact native window, same as any other app.

[![CI](https://github.com/manofculture55/Nexus-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/manofculture55/Nexus-Agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Launch NEXUS and ask it to summarise a PDF, check your battery, rename a file, search your documents, or answer a question — and it does the work locally on CPU.

| | |
|---|---|
| **Documents** | Summarise TXT, PDF, XLSX, DOCX — one file or a whole folder |
| **Files** | Rename, copy, move, delete, list — behind a folder allowlist |
| **System** | Battery, running processes, shutdown/restart/sleep |
| **Search** | Semantic search across your own documents (RAG) |
| **Memory** | Remembers facts about you between sessions |
| **Training** *(experimental)* | LoRA fine-tuning on your own Q&A data — see note below |

Fully offline — no internet access required for any feature. Runs on a laptop with **no GPU** and 8 GB of RAM.

---

## Install

```bash
git clone https://github.com/manofculture55/Nexus-Agent.git
cd Nexus-Agent
setup.bat
```

Then download the model (~2 GB) and put it in `model/`:

> [qwen2.5-3b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)

Launch with `quickopen.bat`, or:

```bash
nexus              # console script
python -m nexus    # module entry point
```

> **Note on `llama-cpp-python`:** install it from the prebuilt CPU wheel index (`setup.bat` does this for you). Building from source on Windows fails on the 260-character path limit while unpacking the vendored llama.cpp sources.

---

## Architecture

```
src/nexus/
├── config.py          Single source of truth for paths and settings
├── agent/
│   ├── router.py      Intent classification and dispatch
│   ├── commands.py    Slash commands (/task, /remember, …)
│   └── memory.py      Session context + persistent long-term memory
├── llm/
│   └── loader.py      GGUF model loading, LoRA resolution, inference
├── tools/
│   ├── files.py       Read, summarise, rename, copy, move, delete, save
│   ├── system.py      Battery, processes, power, task queue
│   └── rag.py         Document indexing and vector search
├── security/
│   ├── permissions.py Folder allowlist for file access
│   └── confirm.py     Confirmation gateway for destructive actions
├── ui/
│   ├── server.py       FastAPI backend (HTTP + WebSocket)
│   ├── desktop.py      pywebview window launcher (console-script entry point)
│   ├── web/            HTML/CSS/JS frontend
│   ├── app.py          Tkinter Quick Menu (legacy, being retired)
│   └── themes.py       Dark / Light / Blue theme definitions
└── training/
    └── trainer.py     LoRA fine-tuning pipeline (experimental, see below)
```

**Request flow:** input → slash command? → keyword fast path → LLM intent classification → tool → response.

> **Training is experimental.** `trainer.py` runs real LoRA fine-tuning and
> produces a working HuggingFace PEFT adapter — but NEXUS's inference engine
> (`llama-cpp-python`) only loads GGUF-format adapters. Closing that gap
> means running llama.cpp's `convert_lora_to_gguf.py` on the output by hand
> (that script isn't published as an installable package, so it isn't
> wired into `trainer.py` automatically). Until that conversion step is
> run, training completes successfully but the base model keeps answering
> — the custom knowledge isn't live yet.

### Safety model

Two independent gates protect the machine:

- **`security/permissions.py`** — file operations are confined to an explicit folder allowlist.
- **`security/confirm.py`** — every destructive action (delete, shutdown, restart, reset training) must be approved through a registered frontend prompt. It **fails closed**: with no frontend registered, approval is denied, so scripts and tests can never destroy anything.

---

## Usage

```
summarise notes.pdf                      what is my battery percentage?
summarise all files in D:\project        show running processes
rename notes.txt to old.txt              remember my birthday is 15 March
what do my files say about X?            what do you remember about me
```

**Slash commands:** `/task` `/remember` `/recall` `/forget` `/save` `/model` `/settings` `/theme` `/clear` `/help`

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # run the test suite
ruff check src tests      # lint
```

Optional feature groups: `.[rag]`, `.[training]`, `.[all]`.

---

## Configuration

| File | Purpose |
|---|---|
| `config/settings.json` | Model path, `max_tokens`, `n_ctx`, threads, theme |
| `config/permissions.json` | Folders NEXUS may read and write |

Each has a `.template.json` alongside it. All are editable from **Settings** in the three-dot menu.

---

## License

MIT
