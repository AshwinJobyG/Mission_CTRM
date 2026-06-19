"""MCP server exposing the local knowledge assistant as tools.

Tools:
  - list_files(folder)        : list readable files in a folder
  - read_file(path)           : read a single file's extracted text
  - ingest_folder(folder)     : load + chunk + embed a folder into Chroma
  - search_knowledge(query)   : vector search, returns matching chunks
  - ask(question)             : full RAG answer grounded in the indexed corpus
  - kb_status()               : configuration + vector DB size

Run (stdio transport, for Claude Desktop / MCP clients):
    python mcp_server.py

Configure an MCP client with:
    {
      "mcpServers": {
        "knowledge": { "command": "python", "args": ["mcp_server.py"] }
      }
    }
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app import config, vectorstore
from app.ingest import ingest_folder as _ingest_folder
from app.loader import load_folder
from app.rag import answer_question

mcp = FastMCP("knowledge")


@mcp.tool()
def kb_status() -> str:
    """Show the active configuration and how many chunks are indexed."""
    return config.summary() + f"\nChunks indexed: {vectorstore.count()}"


@mcp.tool()
def list_files(folder: str = "", recursive: bool = True) -> str:
    """List readable files in a folder (defaults to the configured DATA_DIR)."""
    target = folder or str(config.DATA_DIR)
    docs, skipped = load_folder(target, recursive=recursive)
    lines = [f"Folder: {target}", f"Readable files: {len(docs)}"]
    for d in docs:
        lines.append(f"  - {d.source} ({len(d.text)} chars)")
    if skipped:
        lines.append(f"Skipped (binary/empty/unsupported): {len(skipped)}")
    return "\n".join(lines)


@mcp.tool()
def read_file(path: str, max_chars: int = 8000) -> str:
    """Return the extracted text of a single file (truncated to max_chars)."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"[error] Not a file: {p}"
    docs, _ = load_folder(p.parent, recursive=False)
    for d in docs:
        if Path(d.path) == p.resolve():
            text = d.text
            if len(text) > max_chars:
                return text[:max_chars] + f"\n\n[...truncated, {len(text)} chars total]"
            return text
    return f"[error] Could not extract text from: {p}"


@mcp.tool()
def ingest_folder(folder: str = "", reset: bool = False, recursive: bool = True) -> str:
    """Read every file in a folder, embed it, and store it in the vector DB."""
    target = folder or str(config.DATA_DIR)
    report = _ingest_folder(folder=target, recursive=recursive, reset=reset)
    return str(report)


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5) -> str:
    """Vector-search the indexed corpus and return the matching chunks."""
    results = vectorstore.query(query, top_k=top_k)
    if not results:
        return "No matches (is the corpus ingested?)."
    blocks = []
    for i, r in enumerate(results, 1):
        snippet = r.text[:500] + ("..." if len(r.text) > 500 else "")
        blocks.append(f"[{i}] {r.source} (distance {r.distance:.3f})\n{snippet}")
    return "\n\n".join(blocks)


@mcp.tool()
def ask(question: str, top_k: int = 5) -> str:
    """Answer a question grounded in the indexed corpus, with citations."""
    ans = answer_question(question, top_k=top_k)
    return f"{ans.answer}\n\nSources:\n{ans.format_sources()}"


if __name__ == "__main__":
    mcp.run()
