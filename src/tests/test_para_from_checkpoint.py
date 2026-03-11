import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import torch
from argparse import Namespace
from torch.utils.data import DataLoader

from paraphrase_detection import ParaphraseGPT, add_arguments
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase

# ckpt = "best_25-0.001-sonnet.pt"
ckpt = str(_REPO_ROOT / "checkpoints" / "sonnet_gen_full_ft.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved = torch.load(ckpt, map_location=device, weights_only=False)
base_args = saved["args"]
args = Namespace(
    para_train=str(_REPO_ROOT / "data" / "quora-train.csv"),
    para_dev=str(_REPO_ROOT / "data" / "quora-dev.csv"),
    para_test=str(_REPO_ROOT / "data" / "quora-test-student.csv"),
    para_dev_out=str(_REPO_ROOT / "predictions" / "transfer_from_sonnet_para_dev.csv"),
    para_test_out=str(_REPO_ROOT / "predictions" / "transfer_from_sonnet_para_test.csv"),
    batch_size=8,
    model_size=base_args.model_size,
    use_gpu=(device.type == "cuda"),
)
args = add_arguments(args)

model = ParaphraseGPT(args).to(device)
gpt_state = {
    k[len("gpt.") :]: v for k, v in saved["model"].items() if k.startswith("gpt.")
}
model.gpt.load_state_dict(
    gpt_state, strict=False
)  # loads sonnet-tuned backbone; head stays random
model.eval()

dev_data = ParaphraseDetectionDataset(load_paraphrase_data(args.para_dev), args)
test_data = ParaphraseDetectionTestDataset(
    load_paraphrase_data(args.para_test, split="test"), args
)

dev_loader = DataLoader(
    dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=dev_data.collate_fn
)
test_loader = DataLoader(
    test_data,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=test_data.collate_fn,
)

dev_acc, dev_f1, dev_pred, _, dev_ids = model_eval_paraphrase(dev_loader, model, device)
test_pred, test_ids = model_test_paraphrase(test_loader, model, device)

print(f"Dev acc: {dev_acc:.4f}, Dev f1: {dev_f1:.4f}")

with open(args.para_dev_out, "w") as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for i, p in zip(dev_ids, dev_pred):
        f.write(f"{i}, {p}\n")

with open(args.para_test_out, "w") as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for i, p in zip(test_ids, test_pred):
        f.write(f"{i}, {p}\n")

print("Wrote:", args.para_dev_out, args.para_test_out)
