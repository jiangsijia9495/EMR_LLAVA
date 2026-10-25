# Educational local LLaVA document-processing prototype

**Not the original project.** This is a newly written reference implementation to learn what an EMR-to-research-data pipeline *could* look like. No original dataset, code, model checkpoint ID, or research-variable definitions have been provided. Do not claim this was the original implementation or that you built the original system.

## Security / limitations

- Use **synthetic or authorised de-identified data only**. Before using real EMRs, secure organisational and legal approval, access controls, appropriate retention policies, and a privacy assessment.
- Files stay on the local machine for inference after checkpoint installation; downloading a checkpoint may require internet. No file is sent to a remote LLaVA API by this code.
- This demo does NOT perform OCR, diagnose medical images, process DICOM, identify clinical variables, link patient encounters, or produce a validated regression-ready patient dataset.
- LLaVA 1.5 is not a medical diagnostic model or a reliable source of exact values in tiny document text or difficult handwriting.
- The generated model description is not written to the CSV; do not log or export it with sensitive source data. The generated labels themselves still require review.

## Files

- `pipeline.py`: file format routing, PDF text/table extraction, model inference, JSON validation, page-level metadata CSV.
- `make_demo_data.py`: constructs synthetic PDF, CSV, TXT, and image inputs. No real medical information.
- `inspect_patch_features.py`: optional GPU exercise to inspect projected patch tensors returned by LLaVA.
- `tests/test_pipeline.py`: tests parsing and JSON validation without GPU.
- `requirements.txt`: minimum packages for the sample.

## Running on Ubuntu + RTX 4090

Install an NVIDIA driver and a PyTorch build matching your CUDA environment (see official PyTorch install selector). Use Python 3.10+ in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install CUDA-enabled PyTorch first (command from https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
python make_demo_data.py
python pipeline.py --input demo_input --output demo_output --no-model
python -m unittest discover -s tests -v
# If GPU and checkpoint are available:
python pipeline.py --input demo_input --output demo_output
# Optional: examine visual patch representations (NOT automatically research features)
python inspect_patch_features.py demo_input/sample_scan.png
```

GPU code loads `llava-hf/llava-1.5-7b-hf` once in FP16 (`AutoProcessor` and `LlavaForConditionalGeneration`). The full model includes the vision backbone, projector, and language model; it is **not** necessary to separately load all three components for ordinary inference. A 24 GB RTX 4090 can often run this checkpoint with low batch sizes, but actual memory depends on environment and generation length. For CUDA out-of-memory errors, reduce concurrency/length or consider 4-bit model quantisation (not included in this demo).

## Data flow

1. Enumerate file types rather than assume that every item is an X-ray, MRI, PDF, or table.
2. Extract digital PDF text and (when possible) vector-PDF tables with PyMuPDF; read CSV as an actual table; read text files separately.
3. Render PDF pages to images; use LLaVA only for visual layout classification and likely handwriting, not precise measurement extraction.
4. Parse and validate JSON responses; mark uncertain/invalid predictions for review.
5. Export one **document/page row**, including document identifier, number of characters, tables, predicted page kind, and review flag. Output is **not regression-ready until the researchers define variables, units, outcomes, and the unit of observation**.

### What to add for a real research pipeline

- A defined schema from researchers: e.g., `observation_id`, `X1`, `X2`, `Y` and rules for missing values.
- Dedicated OCR/table extraction for image-based pages, with source bounding boxes and traceability.
- Approved linkage of pages/files to observation IDs; page count must not be confused with patient count.
- Deterministic normalisation for numerical fields and units, date parsing, categorical mappings, duplicate checks, and quality assurance against a held-out human-reviewed reference set.
- Privacy controls, encryption, redaction, access control, failure/retry logic, audit logs (without sensitive text), reproducible model versions, and quantitative benchmarks.

