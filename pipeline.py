#!/usr/bin/env python3
"""PipeStone - facade layout PDF analyzer. Entry point wrapper."""

from pipeline_logic import APP_NAME, DEFAULT_DPI, DEFAULT_OUTPUT_DIR, analyze_image_file, analyze_pdf_file, setup_logging

# Re-export for backward compatibility
__all__ = ["analyze_pdf_file", "analyze_image_file", "setup_logging", "APP_NAME", "DEFAULT_DPI", "DEFAULT_OUTPUT_DIR"]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--pdf", required=False)
    parser.add_argument("--image", required=False)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ocr-backend", default="auto")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if not args.pdf and not args.image:
        parser.print_help()
        exit(2)
    if args.pdf and args.image:
        parser.error("Use either --pdf or --image, not both")

    setup_logging()
    if args.image:
        result = analyze_image_file(args.image, output_dir=args.output_dir, ocr_backend=args.ocr_backend)
    else:
        result = analyze_pdf_file(
            args.pdf,
            dpi=args.dpi,
            output_dir=args.output_dir,
            ocr_backend=args.ocr_backend,
            force_ocr=args.force_ocr,
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
