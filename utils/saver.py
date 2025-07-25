from pathlib import Path
import torch
import pandas as pd

def save_test_outputs(run_tag: str,
                     logits,          # Tensor (N, C)
                     labels,          # Tensor (N,)
                     img_paths=None) -> Path:
    """
    Save logits+labels as .pt  (+ optional CSV) and return the .pt path.
    """
    root = Path("saved_models")
    root.mkdir(exist_ok=True)

    pt_path = root / f"{run_tag}_val_logits.pt"
    torch.save({"logits": logits.cpu(),
                "labels": labels.cpu()}, pt_path)

    # optional CSV for EDA
    if img_paths is not None:
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(1)
        df = pd.DataFrame({
            "sample_id"        : [Path(p).stem for p in img_paths],
            "true_label"       : labels.cpu().numpy(),
            "pred_label"       : preds,
            "logit_normal"     : logits[:, 0].cpu().numpy(),
            "logit_osteopenia" : logits[:, 1].cpu().numpy(),
            "logit_osteoporosis": logits[:, 2].cpu().numpy(),
            "prob_normal"      : probs[:, 0],
            "prob_osteopenia"  : probs[:, 1],
            "prob_osteoporosis": probs[:, 2],
        })
        df["correct"] = (df["true_label"] == df["pred_label"]).astype(int)
        csv_path = pt_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    return pt_path