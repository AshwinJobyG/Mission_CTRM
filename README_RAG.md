# Local RAG Knowledge Assistant (PS-019 PoC)

Read any files from a folder → embed them into a local **Chroma** vector DB →
ask questions and get **source-cited answers** from a **locally hosted LLM
(Ollama)**. Exposed three ways: an **MCP server**, a **CLI**, and a **web UI**.

```
folder of files ──► loader ──► chunker ──► Ollama embeddings ──► Chroma (vector DB)
                                                                      │
your prompt ──► retrieve top-k context ──► Ollama LLM (grounded) ──► answer + citations
```

## 1. Prerequisites

Install [Ollama](https://ollama.com) and pull a chat + embedding model:

```bash
ollama serve            # starts the local server on :11434
ollama pull llama3              # chat model
ollama pull nomic-embed-text   # embedding model
```

Install Python dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) copy `.env.example` to `.env` to change models/paths.

## 2. Ingest a folder

Drop your documents into `./data` (sample files are included), then:

```bash
python -m app.cli ingest                 # ingest ./data
python -m app.cli ingest /path/to/folder # any folder
python -m app.cli ingest --reset         # rebuild from scratch
```

Supported files: txt, md, pdf, docx, csv/tsv, json, html, and most code/text
formats. Binary/media files are skipped automatically.

## 3. Ask questions

**CLI (one-shot):**
```bash
python -m app.cli ask "What caused incident INC-2041?"
```

**CLI (interactive):**
```bash
python -m app.cli chat
```

**Web UI:**
```bash
streamlit run app/web.py
```
Ingest from the sidebar, then chat. Each answer shows its sources.

## 4. MCP server

```bash
python mcp_server.py     # stdio transport
```

Register it with an MCP client (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "knowledge": { "command": "python", "args": ["/abs/path/mcp_server.py"] }
  }
}
```

Tools exposed: `list_files`, `read_file`, `ingest_folder`, `search_knowledge`,
`ask`, `kb_status`.

## 5. How it maps to PS-019

| PS-019 requirement | Where |
|--------------------|-------|
| Read enterprise files (multi-source) | `app/loader.py` |
| Vector DB for the data | `app/vectorstore.py` (Chroma) |
| Local LLM for context/generation | `app/rag.py` + `app/embeddings.py` (Ollama) |
| Source-backed, no-hallucination answers | `SYSTEM_PROMPT` in `app/rag.py` |
| Prompt → grounded answer | CLI / Web / MCP `ask` |

## Configuration

All knobs live in `app/config.py` and are overridable via env vars — see
`.env.example` (Ollama host, chat/embed model, chunk size, top-k, paths).
