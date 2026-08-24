#!/usr/bin/env python3
"""PipeStone - facade layout PDF analyzer. Entry point wrapper."""

from pipeline_logic import APP_NAME, DEFAULT_DPI, DEFAULT_OUTPUT_DIR, analyze_image_file, analyze_pdf_file, setup_logging
from pipestone_ocr import DEFAULT_TESSERACT_LANGUAGE

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
    parser.add_argument("--ocr-language", default=DEFAULT_TESSERACT_LANGUAGE)
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument(
        "--calculate-area",
        action="store_true",
        help="Calculate facade area on pages containing the legend hatch pattern.",
    )
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if not args.pdf and not args.image:
        parser.print_help()
        exit(2)
    if args.pdf and args.image:
        parser.error("Use either --pdf or --image, not both")

    setup_logging()
    if args.image:
        result = analyze_image_file(
            args.image,
            output_dir=args.output_dir,
            ocr_backend=args.ocr_backend,
            tesseract_language=args.ocr_language,
        )
    else:
        result = analyze_pdf_file(
            args.pdf,
            dpi=args.dpi,
            output_dir=args.output_dir,
            ocr_backend=args.ocr_backend,
            tesseract_language=args.ocr_language,
            force_ocr=args.force_ocr,
            calculate_area=args.calculate_area,
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.pdf:
        if args.calculate_area:
            print(
                json.dumps(
                    {
                        "hatch_pages": result.get("hatch_pages", []),
                        "area_calculation": result.get("area_calculation", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(result.get("hatch_pages", []), ensure_ascii=False))
