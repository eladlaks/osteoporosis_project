# ensembles/run_ensemble.py
import sys, pathlib, argparse, ast, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataset_handler.dataset import ImageDataset
from ensembles.soft_voting     import SoftVotingEnsemble
from ensembles.weighted_voting import WeightedVotingEnsemble

# ---------- helpers ----------
def _as_list(s: str):
    return ast.literal_eval(s) if s.startswith("[") else [x.strip() for x in s.split(",")]

def build_ensemble(args, device):
    ckpts = _as_list(args.ckpts)
    archs = _as_list(args.archs)
    if args.type == "soft":
        return SoftVotingEnsemble(ckpts, archs, device)
    if args.type == "weighted":
        weights = _as_list(args.weights)
        return WeightedVotingEnsemble(ckpts, archs, weights, device)
    sys.exit(f"Unknown ensemble type {args.type}")

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--type",    required=True, choices=["soft", "weighted"])
p.add_argument("--ckpts",   required=True)
p.add_argument("--archs",   required=True)
p.add_argument("--weights", default=None)
p.add_argument("--data",    default="data/test_cropped_data")
p.add_argument("--batch",   type=int, default=32)
args = p.parse_args()

device   = "cuda" if torch.cuda.is_available() else "cpu"
ensemble = build_ensemble(args, device).eval().to(device)

loader = DataLoader(ImageDataset(args.data),
                    batch_size=args.batch, shuffle=False)

all_true, all_pred, all_prob, all_path = [], [], [], []
with torch.no_grad():
    for imgs, labels, paths in loader:
        probs = ensemble(imgs.to(device)).softmax(1)  # (B,C)
        preds = probs.argmax(1).cpu()
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.cpu().tolist())
        all_path.extend(paths)

# -------- metrics --------
acc = (torch.tensor(all_true) == torch.tensor(all_pred)).float().mean().item()
cm  = confusion_matrix(all_true, all_pred)

# -------- save CSV --------
csv_name = f"{args.type}_preds_{datetime.now():%Y%m%d_%H%M%S}.csv"
cols = [f"p{i}" for i in range(len(all_prob[0]))]
df = pd.DataFrame({
    "sample": all_path,
    "true":   all_true,
    "pred":   all_pred,
    "conf":   [max(p) for p in all_prob],
    **{c: [row[i] for row in all_prob] for i, c in enumerate(cols)}
})
df.to_csv(csv_name, index=False)

# -------- confusion-matrix figure --------
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", colorbar=False)
fig_name = csv_name.replace(".csv", "_cm.png")
plt.savefig(fig_name, dpi=150)
plt.close()

print(f"ensemble_acc = {acc:.4f}  |  CSV → {csv_name}")

# -------- optional wandb logging --------
try:
    import wandb
    run = wandb.init(project="final_project", job_type="ensemble",
                     config=vars(args))
    wandb.log({"ensemble_acc": acc})
    art = wandb.Artifact(f"{args.type}_results", type="predictions")
    art.add_file(csv_name)
    art.add_file(fig_name)
    run.log_artifact(art)
    run.finish()
except ImportError:
    pass