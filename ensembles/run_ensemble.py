# ensemble/run_ensemble.py
"""
Run an already-trained ensemble (soft / weighted / stacking) on a dataset.

"""

import argparse, ast, torch, sys
from torch.utils.data import DataLoader
from pathlib import Path

from dataset_handler.dataset import ImageDataset
from ensembles.soft_voting    import SoftVotingEnsemble
from ensembles.weighted_voting import WeightedVotingEnsemble
from ensembles.stacking        import StackingEnsemble


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def _as_list(s: str):
    """Convert '[1,2]' or 'a,b' into list."""
    return ast.literal_eval(s) if "[" in s else [x.strip() for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True,
                   choices=["soft", "weighted", "stacking"],
                   help="Ensemble method")
    p.add_argument("--ckpts", required=True,
                   help="Comma-sep or Python list of checkpoint paths")
    p.add_argument("--archs", required=True,
                   help="Comma-sep or Python list of arch names")
    p.add_argument("--weights", default=None,
                   help="Weights list (only for weighted)")
    p.add_argument("--meta", default=None,
                   help="meta_clf.pkl (only for stacking)")
    p.add_argument("--data", default="data/test_cropped_data",
                   help="Directory with test images")
    p.add_argument("--batch", type=int, default=32)
    return p.parse_args()


# ------------------------------------------------------------------ #
# Build ensemble
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try to start a W&B run (optional)
    try:
        import wandb
        run = wandb.init(project="final_project",
                         job_type="ensemble",
                         config=vars(args))
    except ImportError:
        wandb = None
        run = None

    ensemble = build_ensemble(args, device).eval().to(device)

    loader = DataLoader(
        ImageDataset(args.data),
        batch_size=args.batch, shuffle=False)

    correct = total = 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            preds = ensemble(imgs.to(device)).argmax(1)
            correct += (preds.cpu() == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    print(f"Ensemble accuracy = {acc:.4f}")

    if wandb:
        wandb.log({"ensemble_acc": acc})
        import pandas as pd
        from pathlib import Path
        from datetime import datetime

        # ---------- save predictions CSV ----------
        csv_name = f"{args.type}_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(
            {"true": all_labels, "pred": all_preds}
        )
        df.to_csv(csv_name, index=False)

        if wandb:
            art = wandb.Artifact(f"{args.type}_preds", type="predictions")
            art.add_file(csv_name)
            wandb.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()