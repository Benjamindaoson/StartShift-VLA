# Offline archive audit

Run `python scripts/verify_archive.py` from the repository root. This reads the saved records and verifies metrics, frozen group labels, seed schedules and exported hashes. It never trains, simulates or downloads data.

The prior `startshift/research_audit.py` contains additional pure audit utilities and a metadata CLI. Its metadata CLI expects private dataset Parquet files and the original output layout; it is retained as incomplete audit work, not the public quick-start command.
