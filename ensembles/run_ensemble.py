# ensembles/run_ensemble.py
"""
Run an already-trained ensemble (soft / weighted / stacking) on a dataset.
"""

# Make sure project root is importable when executed via wandb agent
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import argparse, ast, torch
from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd

from dataset_handler.dataset import ImageDataset
from ensembles.soft_voting      import SoftVotingEnsemble
from ensembles.weighted_voting  import WeightedVotingEnsemble
from ensembles.stacking         import StackingEnsemble


# -------------- CLI --------------
def _as_list(s: str):
    """Convert '[1,2]' or 'a,b' into list form."""
    return ast.literal_eval(s) if "[" in s else [x.strip() for x in s.split(",")]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--type",    required=True, choices=["soft", "weighted", "stacking"])
    p.add_argument("--ckpts",   required=True)
    p.add_argument("--archs",   required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--meta",    default=None)
    p.add_argument("--data",    default="data/test_cropped_data")
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--num_classes", type=int, default=3,
                   help="Number of target classes (needed by model builders)")
    return p.parse_args()


# ---------- ensemble builder ----------
def build_ensemble(args, device):
    ckpts = _as_list(args.ckpts)
    archs = _as_list(args.archs)
    if args.type == "soft":
        return SoftVotingEnsemble(ckpts, archs, device)
    if args.type == "weighted":
        weights = _as_list(args.weights)
        assert len(weights) == len(ckpts), "weights length mismatch"
        return WeightedVotingEnsemble(ckpts, archs, weights, device)
    if args.type == "stacking":
        if not args.meta:
            sys.exit("Stacking requires --meta path")
        return StackingEnsemble(ckpts, archs, args.meta, device)
    sys.exit(f"Unknown ensemble type {args.type}")


# ----------------- main -----------------
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # optional wandb run
    try:
        import wandb
        run = wandb.init(project="final_project", job_type="ensemble",
                         config=vars(args))
        # ensure NUM_CLASSES is available for model builders
        if "NUM_CLASSES" not in wandb.config:
            wandb.config.update({"NUM_CLASSES": args.num_classes},
                                allow_val_change=True)
    except ImportError:
        wandb = None
        run   = None

    ensemble = build_ensemble(args, device).eval().to(device)

    loader = DataLoader(ImageDataset(args.data),
                        batch_size=args.batch, shuffle=False)

    correct = total = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            preds = ensemble(imgs.to(device)).argmax(1)
            correct += (preds.cpu() == labels).sum().item()
            total   += labels.size(0)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = correct / total
    print(f"Ensemble accuracy = {acc:.4f}")

    if wandb:
        wandb.log({"ensemble_acc": acc})

        # save predictions CSV
        csv_name = f"{args.type}_preds_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame({"true": all_labels, "pred": all_preds}).to_csv(csv_name, index=False)

        art = wandb.Artifact(f"{args.type}_preds", type="predictions")
        art.add_file(csv_name)
        wandb.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()