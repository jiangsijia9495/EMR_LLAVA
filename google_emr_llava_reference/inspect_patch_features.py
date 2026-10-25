"""Optional GPU-only exercise: inspect patch-level projected LLaVA features.

These are the selected CLIP patch representations *after the LLaVA projector*,
not independently trained clinical regression features. Not used in pipeline.py.
Requires the model checkpoint and supported Transformers version.
"""

import argparse
from PIL import Image
from pipeline import LlavaInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a non-sensitive PNG/JPG image")
    args = parser.parse_args()

    inf = LlavaInference()
    with Image.open(args.image) as img:
        pixels = inf.processor.image_processor(
            images=img.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(inf.model.device, dtype=inf.torch.float16)

    with inf.torch.inference_mode():
        features = inf.model.get_image_features(pixel_values=pixels)

    # Version-dependent API: older models return a list of per-image tensors.
    if isinstance(features, (list, tuple)):
        features = features[0]
    elif hasattr(features, "pooler_output"):
        features = features.pooler_output
    print("Projected patch representation shape:", tuple(features.shape))
    print("These are NOT automatically meaningful research variables.")


if __name__ == "__main__":
    main()
