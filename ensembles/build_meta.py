# ensemble/build_meta.py
"""
Train a meta-classifier (LogisticRegression) on validation logits that were
saved earlier by save_val_outputs(). Produces meta_clf.pkl for stacking.

Example
-------
python ensemble/build_meta.py \
    --logits "saved_models/runA_ResNet50_best_val_logits.pt,\
              saved_models/runB_VGG19_best_val_logits.pt" \
    --out    "ensemble_assets/meta_clf.pkl"
"""

import argparse, glob, joblib, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--logits",
        required=True,
        help="Comma-separated list (or glob) of *_val_logits.pt files",
    )
    p.add_argument(
        "--out",
        default="ensemble_assets/meta_clf.pkl",
        help="Output path for the pickled meta-classifier",
    )
    p.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for LogisticRegression",
    )
    return p.parse_args()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args = parse_args()

    # ---------- collect *.pt files ----------------------------------
    if "*" in args.logits:                    # allow glob pattern
        pt_files = glob.glob(args.logits)
    else:
        pt_files = [p.strip() for p in args.logits.split(",")]
    assert pt_files, "No logits files found"

    # ---------- load tensors and build design matrix ---------------
    records = [torch.load(p) for p in pt_files]
    X = torch.cat([r["logits"] for r in records], dim=1).numpy()  # (N, M·C)
    y = records[0]["labels"].numpy()                              # (N,)

    # ---------- train meta-learner ---------------------------------
    clf = LogisticRegression(
        max_iter=args.max_iter, multi_class="multinomial"
    )
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"Meta train accuracy (on val set) = {acc:.3f}")

    # ---------- save to disk ---------------------------------------
    out_p = Path(args.out)
    out_p.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, out_p)
    print(f"Saved meta-classifier → {out_p.resolve()}")

    # ---------- optional: upload as W&B artifact -------------------
    try:
        import wandb

        run = wandb.init(project="final_project", job_type="meta_training")
        art = wandb.Artifact("meta_clf", type="meta-model")
        art.add_file(str(out_p))
        run.log_artifact(art, aliases=["latest", "best"])
        run.finish()
        print("Uploaded meta_clf.pkl as W&B artifact")
    except ImportError:
        pass


if __name__ == "__main__":
    main()