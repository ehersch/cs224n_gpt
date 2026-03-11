"""
Test sonnet score: load a local checkpoint, run inference on held-out prompts on Modal,
then compute chrF against the gold file.

Usage:
  modal run test_sonnet_score.py --checkpoint checkpoints/19_20-1e-05-sonnet.pt
  modal run test_sonnet_score.py --checkpoint checkpoints/my_model.pt --temperature 1.2 --top-p 0.9

The checkpoint path is a local file path (relative or absolute). The file is read
locally and sent to Modal for inference, so no need to bake it into the image.
"""

import os
import sys
from pathlib import Path
import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if modal is not None:
    MODAL_WORKSPACE = "/workspace"
    MODAL_SRC = os.path.join(MODAL_WORKSPACE, "src")
    HELD_OUT_PATH = "data/sonnets_held_out.txt"
    GOLD_PATH = "data/TRUE_sonnets_held_out.txt"

    sonnet_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers",
            "einops",
            "tqdm",
            "requests",
            "filelock",
            "importlib-metadata",
            "sacrebleu",
            "scikit-learn",
        )
        .workdir(MODAL_WORKSPACE)
        .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
    )

    app = modal.App("test-sonnet-score")

    @app.function(image=sonnet_image, gpu="L4", timeout=600)
    def run_inference(
        checkpoint_bytes: bytes,
        held_out_path: str = HELD_OUT_PATH,
        temperature: float = 1.2,
        top_p: float = 0.9,
    ) -> str:
        """Load checkpoint from uploaded bytes, run generation on held-out prompts, return generated file content."""
        if MODAL_SRC not in sys.path:
            sys.path.insert(0, MODAL_SRC)
        import tempfile
        import torch
        from sonnet_generation import add_arguments, SonnetGPT
        from datasets import SonnetsDataset

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(checkpoint_bytes)
            full_checkpoint_path = f.name
        try:
            saved = torch.load(full_checkpoint_path, weights_only=False, map_location="cuda")
        finally:
            os.unlink(full_checkpoint_path)

        full_held_out = os.path.join(MODAL_WORKSPACE, held_out_path)
        args = add_arguments(saved["args"])
        model = SonnetGPT(args)
        model.load_state_dict(saved["model"])
        model = model.to("cuda")
        model.eval()

        dataset = SonnetsDataset(full_held_out)
        generated_sonnets = []
        for batch in dataset:
            sonnet_id = batch[0]
            encoding = model.tokenizer(
                batch[1], return_tensors="pt", padding=False, truncation=True
            ).to("cuda")
            output = model.generate(
                encoding["input_ids"], temperature=temperature, top_p=top_p
            )[0][0]
            decoded = model.tokenizer.decode(output, skip_special_tokens=True)
            generated_sonnets.append((sonnet_id, f"{decoded}\n\n"))

        lines = ["--Generated Sonnets-- \n\n"]
        for sonnet in generated_sonnets:
            lines.append(f"\n{sonnet[0]}\n")
            lines.append(sonnet[1])
        return "".join(lines)

    @app.local_entrypoint()
    def main(
        checkpoint: str = "checkpoints/19_20-1e-05-sonnet.pt",
        held_out_path: str = HELD_OUT_PATH,
        gold_path: str = GOLD_PATH,
        temperature: float = 1.2,
        top_p: float = 0.9,
        out_path: str = None,
    ):
        """Run inference on Modal, save generated file, compute chrF vs gold, and print score."""
        if out_path is None:
            out_path = str(_REPO_ROOT / "predictions" / "generated_sonnets_held_out.txt")
        if not checkpoint:
            print("Error: --checkpoint is required (e.g. checkpoints/19_20-1e-05-sonnet.pt)")
            sys.exit(1)
        if not os.path.isfile(checkpoint):
            print(f"Error: checkpoint file not found: {os.path.abspath(checkpoint)}")
            sys.exit(1)
        print(f"Checkpoint: {checkpoint} (local)")
        print(f"Held-out prompts: {held_out_path}")
        print(f"Gold reference: {gold_path}")
        print("Uploading checkpoint and running inference on Modal...")
        with open(checkpoint, "rb") as f:
            checkpoint_bytes = f.read()
        submission_txt = run_inference.remote(
            checkpoint_bytes=checkpoint_bytes,
            held_out_path=held_out_path,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(submission_txt)
        print(f"Generated file saved to {out_path}")

        from evaluation import test_sonnet

        chrf_score = test_sonnet(out_path, gold_path)
        print(f"chrF score: {chrf_score}")
