# Logging System - Decisions

## 2026-04-10
- Using Python stdlib logging (4 files already have it)
- Logger name: "steering_geometry" (named logger, not root)
- Dual output: StreamHandler (console) + FileHandler (file)
- Log file naming: steering_YYYYMMDD_HHMMSS.log
- Log dir: logs/ (root)
- Idempotent configure_logging() to avoid duplicate handlers
- Lazy %s format (matching stability_comparison.py gold standard)
