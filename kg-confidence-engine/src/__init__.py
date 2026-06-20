"""kg-confidence-engine source package."""

# Load a project-root .env (if python-dotenv is installed) so credentials in
# .env are picked up by every entry point without an explicit export.
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv optional / .env absent — fall back to os.environ
    pass
