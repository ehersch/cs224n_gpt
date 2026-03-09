"""
Generate dev and test predictions from a hyperparam_sweep checkpoint.
"""
import torch
from torch.utils.data import DataLoader

from hyperparam_sweep import ParaphraseGPT, add_arguments
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase

CKPT = "sweep_results/w6_medium_bs16_15ep_fp16.pt"
GPU = 0
DEV_OUT = "predictions/para-dev-output.csv"
TEST_OUT = "predictions/para-test-output.csv"

device = torch.device(f"cuda:{GPU}")

print(f"Loading checkpoint: {CKPT}")
saved = torch.load(CKPT, map_location=device, weights_only=False)
args = saved["args"]

# Override paths and device
args.para_train = "data/quora-train.csv"
args.para_dev = "data/quora-dev.csv"
args.para_test = "data/quora-test-student.csv"
args.use_gpu = True
args.gpu = GPU

args = add_arguments(args)

model = ParaphraseGPT(args)
model.load_state_dict(saved["model"])
model = model.to(device)
model.eval()

print(f"Model: {args.model_size}, d={args.d}, l={args.l}")

# Load datasets
dev_data = ParaphraseDetectionDataset(load_paraphrase_data(args.para_dev), args)
test_data = ParaphraseDetectionTestDataset(
    load_paraphrase_data(args.para_test, split="test"), args
)

dev_loader = DataLoader(
    dev_data, shuffle=False, batch_size=16, collate_fn=dev_data.collate_fn
)
test_loader = DataLoader(
    test_data, shuffle=False, batch_size=16, collate_fn=test_data.collate_fn
)

# Dev predictions
print("Running dev evaluation...")
dev_acc, dev_f1, dev_pred, _, dev_ids = model_eval_paraphrase(dev_loader, model, device)
print(f"Dev acc: {dev_acc:.4f}, Dev f1: {dev_f1:.4f}")

# Test predictions
print("Running test predictions...")
test_pred, test_ids = model_test_paraphrase(test_loader, model, device)

# Write dev predictions
with open(DEV_OUT, "w") as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for i, p in zip(dev_ids, dev_pred):
        f.write(f"{i}, {p}\n")
print(f"Wrote dev predictions: {DEV_OUT}")

# Write test predictions
with open(TEST_OUT, "w") as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for i, p in zip(test_ids, test_pred):
        f.write(f"{i}, {p}\n")
print(f"Wrote test predictions: {TEST_OUT}")

print("Done! Predictions ready for submission.")
