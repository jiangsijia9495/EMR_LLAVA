"""Educational EMR-like document pipeline. Use synthetic/non-sensitive data only.

This is a newly constructed reference, NOT a recovery of anyone's original system.
PDF digital text and tables are parsed deterministically; LLaVA labels page images.
The output is a document/page-level feature prototype, NOT a clinical-grade or
research-ready patient-level dataset. Review and define the research unit of analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
from PIL import Image

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SUPPORTED = {".pdf", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
ALLOWED_KINDS = {"form", "report", "table", "chart", "medical_image", "other", "unknown"}

# A VLM is not a dependable source of exact medical measurements or diagnoses.
VISION_TASK = '''Describe the VISUAL LAYOUT of this document/page, not its medical meaning.
Do not identify people, read exact numeric measurements, or infer a diagnosis.
Reply with ONLY a JSON object. page_kind MUST be exactly one of:
form, report, table, chart, medical_image, other, unknown.
Use these exact keys: page_kind, has_handwriting, brief_visual_description,
needs_human_review. Example:
{"page_kind":"form","has_handwriting":null,
 "brief_visual_description":"A form with printed fields",
 "needs_human_review":true}
Use null instead of true/false for has_handwriting if uncertain.
Set needs_human_review to true when text is difficult to read, handwriting
might be present, content is ambiguous, or you are not confident.'''


def parse_visual_response(answer: str) -> dict[str, Any]:
    """Validate an untrusted generated response. No self-reported confidence used."""
    try:
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end < start:
            raise ValueError("JSON object not found")
        payload = json.loads(answer[start : end + 1])
        kind = payload.get("page_kind")
        handwriting = payload.get("has_handwriting")
        review = payload.get("needs_human_review")
        desc = payload.get("brief_visual_description")
        if kind not in ALLOWED_KINDS:
            raise ValueError("unrecognised page_kind")
        if handwriting is not None and not isinstance(handwriting, bool):
            raise ValueError("has_handwriting must be bool or null")
        if not isinstance(review, bool):
            raise ValueError("needs_human_review must be bool")
        if not isinstance(desc, str):
            raise ValueError("description must be a string")
        return {
            "page_kind": kind,
            "has_handwriting": handwriting,
            "needs_human_review": review or handwriting is None,
            "visual_description": desc[:160],
            "inference_status": "ok",
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {
            "page_kind": "unknown",
            "has_handwriting": None,
            "needs_human_review": True,
            "visual_description": "",
            "inference_status": "invalid_model_output",
        }


class LlavaInference:
    """Lazy-load the complete LLaVA checkpoint ONCE for the whole run."""

    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable. Use --no-model for a CPU parsing demo.")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        )
        self.model.eval()

    def classify(self, image: Image.Image) -> dict[str, Any]:
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": VISION_TASK}
        ]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, images=image.convert("RGB"), return_tensors="pt"
        ).to(self.model.device, self.torch.float16)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=180)
        # Decode NEW text only, not the repeated input prompt.
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return parse_visual_response(answer)


def empty_visual(status: str = "skipped") -> dict[str, Any]:
    return {
        "page_kind": "unknown",
        "has_handwriting": None,
        "needs_human_review": True,
        "visual_description": "",
        "inference_status": status,
    }


def infer_visual(image: Image.Image, llava: LlavaInference | None) -> dict[str, Any]:
    if llava is None:
        return empty_visual()
    try:
        return llava.classify(image)
    except RuntimeError:
        # An OOM or other inference failure should not be silently treated as success.
        return empty_visual("model_runtime_error")


def base_row(doc_id: str, page: int, ext: str) -> dict[str, Any]:
    return {
        "document_id": doc_id, "page_number": page, "file_type": ext,
        "digital_text_chars": 0, "detected_tables": 0,
        "tabular_rows": 0, "tabular_columns": 0,
    }


def process_file(path: Path, doc_id: str, llava: LlavaInference | None) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        rows = []
        with pymupdf.open(path) as pdf:
            for page_index, page in enumerate(pdf):
                row = base_row(doc_id, page_index + 1, ext)
                # Digital text: unlike OCR, this does not read pixels or handwriting.
                row["digital_text_chars"] = len(page.get_text("text", sort=True))
                try:
                    # Table detection works for SOME digital PDFs, not all scans.
                    row["detected_tables"] = len(page.find_tables().tables)
                except (AttributeError, ValueError):
                    row["detected_tables"] = 0
                if llava is None:
                    visual = empty_visual()
                else:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    with Image.open(BytesIO(pix.tobytes("png"))) as raw:
                        visual = infer_visual(raw.convert("RGB"), llava)
                rows.append({**row, **visual})
        return rows

    if ext == ".csv":
        # This demo only reads column counts and dimensions, not patient-level values.
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        row = base_row(doc_id, 1, ext)
        row.update({"detected_tables": 1, "tabular_rows": len(table),
                    "tabular_columns": len(table.columns)})
        return [{**row, **empty_visual("not_applicable"), "page_kind": "table"}]

    if ext == ".txt":
        row = base_row(doc_id, 1, ext)
        row["digital_text_chars"] = len(path.read_text(encoding="utf-8"))
        return [{**row, **empty_visual("not_applicable"), "page_kind": "report"}]

    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        with Image.open(path) as raw:
            # Multi-frame TIFF and DICOM series are NOT handled by this demo.
            visual = infer_visual(raw.convert("RGB"), llava)
        return [{**base_row(doc_id, 1, ext), **visual}]

    raise ValueError(f"Unsupported extension: {ext}")


def run(input_dir: Path, output_dir: Path, llava: LlavaInference | None) -> pd.DataFrame:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError("Input directory does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in SUPPORTED)
    rows = []
    for i, path in enumerate(files, start=1):
        # Sequential IDs avoid copying potentially identifying filenames into outputs.
        doc_id = f"doc_{i:04d}"
        try:
            rows.extend(process_file(path, doc_id, llava))
        except Exception as exc:
            print(f"Warning: {doc_id} failed ({type(exc).__name__})", file=sys.stderr)
            rows.append({**base_row(doc_id, 0, path.suffix.lower()),
                         **empty_visual("file_processing_error")})
    results = pd.DataFrame(rows)
    # Do not export generated prose, original text or source paths by default:
    # they could contain identifying health information.
    results.drop(columns=["visual_description"], errors="ignore").to_csv(
        output_dir / "page_features.csv", index=False
    )
    (output_dir / "run_summary.json").write_text(json.dumps({
        "documents": len(files), "output_pages_or_files": len(rows),
        "vlm_enabled": llava is not None,
        "notice": "Prototype metadata only. Not patient-level regression-ready data."
    }, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational EMR-like page feature demo")
    parser.add_argument("--input", type=Path, default=Path("demo_input"))
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--no-model", action="store_true", help="Test parsing without CUDA/model")
    args = parser.parse_args()
    llava = None if args.no_model else LlavaInference()
    df = run(args.input, args.output, llava)
    print(f"Processed {len(df)} page/file rows. Output in {args.output}")


if __name__ == "__main__":
    main()
