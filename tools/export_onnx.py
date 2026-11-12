"""Export the Schnabel historical-recording U-Net to ONNX.

The graph is the one used by the restoration workflow: two stages, STFT
2048/512, no hall. Stage 2 is the denoised spectrogram.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import unet  # noqa: E402


def build(checkpoint: Path):
    args = SimpleNamespace(
        activation="elu",
        use_SAM=True,
        use_fencoding=True,
        num_tfc=3,
        num_stages=3,
        depth=6,
        f_dim=1025,
    )
    model = unet.build_model_denoise(unet_args=args)
    model.load_weights(str(checkpoint))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = build(checkpoint)
    frames = 64
    dummy = np.zeros((1, frames, 1025, 2), np.float32)
    rng = np.random.default_rng(7)
    dummy += rng.normal(0.0, 0.01, dummy.shape).astype(np.float32)
    reference = model(dummy, training=False)
    stage2 = np.asarray(reference[0])

    import tf2onnx

    spec = [tf.TensorSpec([None, None, 1025, 2], tf.float32, name="spectrogram")]
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=17,
        output_path=str(out),
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: dummy}
    exported = session.run(None, feed)
    best = None
    for index, candidate in enumerate(exported):
        error = float(np.max(np.abs(np.asarray(candidate) - stage2)))
        if best is None or error < best[0]:
            best = (error, index, session.get_outputs()[index].name)
    if best is None or best[0] > 1e-3:
        raise SystemExit(f"exported graph does not match TensorFlow: {best}")
    meta = {
        "input": session.get_inputs()[0].name,
        "output": best[2],
        "max_abs_error": best[0],
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {meta_path}")
    print(f"stage2 max abs error {best[0]:.3e} via {best[2]}")


if __name__ == "__main__":
    main()
