#!/usr/bin/env python3
"""PipeStone - facade layout PDF analyzer. Entry point wrapper."""

from pipeline_main import analyze_pdf_file, setup_logging, APP_NAME, DEFAULT_DPI, DEFAULT_OUTPUT_DIR

# Re-export for backward compatibility
__all__ = ["analyze_pdf_file", "setup_logging", "APP_NAME", "DEFAULT_DPI", "DEFAULT_OUTPUT_DIR"]

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--pdf", required=False)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ocr-backend", default="auto")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--fallback-mm-per-px", type=float, default=None)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if not args.pdf:
        parser.print_help()
        exit(2)

    setup_logging()
    result = analyze_pdf_file(
        args.pdf,
        dpi=args.dpi,
        output_dir=args.output_dir,
        ocr_backend=args.ocr_backend,
        force_ocr=args.force_ocr,
        fallback_mm_per_px=args.fallback_mm_per_px,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))