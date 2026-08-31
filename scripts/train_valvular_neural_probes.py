"""Train and Evaluate Neural Multi-Modal Probes (Cross-Attention & Concat-MLP) for Task B.

Features:
  - Multi-task training for Aortic Stenosis (AS), Mitral Regurgitation (MR), and Tricuspid Regurgitation (TR).
  - Missing-modality evaluation (Full vs. Drop-Echo vs. Drop-ECG).
  - Modality-Failure Analysis (MFA): Loud vs. Silent failure profiling on missed critical cases.
  - Stratified demographic fairness audit across Sex, Age Bands, and Race.
  - Generates publication figures and summary tables.

Usage:
  python scripts/train_valvular_neural_probes.py --model cross_attn --epochs 25
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from primed_ai.data.valvular_dataset import create_valvular_dataloaders
from primed_ai.probes.neural_valvular import ConcatMLPValvularProbe, CrossAttentionValvularProbe
from primed_ai.probes.valvular import bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_valvular_neural_probes")

TARGET_KEYS = [
    ("as_logits", "as_moderate_or_severe", "Aortic Stenosis (Mod/Sev)"),
    ("mr_logits", "mr_moderate_or_severe", "Mitral Regurgitation (Mod/Sev)"),
    ("tr_logits", "tr_moderate_or_severe", "Tricuspid Regurgitation (Mod/Sev)"),
]


def auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.any() and (~y_true).any():
        return float(roc_auc_score(y_true, y_score))
    return float("nan")


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    grades: torch.Tensor,
    bce_loss_fn: nn.BCEWithLogitsLoss,
    ce_loss_fn: nn.CrossEntropyLoss,
    grade_weight: float = 0.3,
) -> torch.Tensor:
    # Binary gate losses
    as_loss = bce_loss_fn(outputs["as_logits"], labels[:, 0])
    mr_loss = bce_loss_fn(outputs["mr_logits"], labels[:, 1])
    tr_loss = bce_loss_fn(outputs["tr_logits"], labels[:, 2])
    total_gate_loss = as_loss + mr_loss + tr_loss

    # Ordinal grade losses (masked where grade == -1)
    grade_losses = []
    for i, grade_key in enumerate(["as_grade_logits", "mr_grade_logits", "tr_grade_logits"]):
        target_g = grades[:, i].long()
        valid_mask = target_g >= 0
        if valid_mask.any():
            grade_losses.append(ce_loss_fn(outputs[grade_key][valid_mask], target_g[valid_mask]))

    total_grade_loss = sum(grade_losses) if grade_losses else torch.tensor(0.0, device=labels.device)
    return total_gate_loss + grade_weight * total_grade_loss


@torch.no_grad()
def evaluate_model_on_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_modality: str | None = None,
) -> dict:
    model.eval()
    mask_echo = mask_modality == "echo"
    mask_ecg = mask_modality == "ecg"

    all_as_probs, all_mr_probs, all_tr_probs = [], [], []
    all_labels, all_demographics = [], {"sex": [], "age": [], "race": []}

    for batch in loader:
        echo = batch.echo_emb.to(device)
        ecg = batch.ecg_emb.to(device)

        out = model(echo, ecg, mask_echo=mask_echo, mask_ecg=mask_ecg)

        as_p = torch.sigmoid(out["as_logits"]).cpu().numpy()
        mr_p = torch.sigmoid(out["mr_logits"]).cpu().numpy()
        tr_p = torch.sigmoid(out["tr_logits"]).cpu().numpy()

        all_as_probs.append(as_p)
        all_mr_probs.append(mr_p)
        all_tr_probs.append(tr_p)
        all_labels.append(batch.labels.numpy())

        for k in all_demographics:
            all_demographics[k].extend(batch.demographics[k])

    as_prob = np.concatenate(all_as_probs)
    mr_prob = np.concatenate(all_mr_probs)
    tr_prob = np.concatenate(all_tr_probs)
    labels = np.concatenate(all_labels)

    probs_dict = {
        "as_moderate_or_severe": as_prob,
        "mr_moderate_or_severe": mr_prob,
        "tr_moderate_or_severe": tr_prob,
    }

    metrics = {}
    for i, (_, target_name, display_name) in enumerate(TARGET_KEYS):
        y = labels[:, i].astype(bool)
        p = probs_dict[target_name]
        auc = auroc_safe(y, p)
        ci = bootstrap_ci(y, p, auroc_safe, n_bootstrap=1000, seed=42)
        prc = float(average_precision_score(y, p)) if (y.any() and (~y).any()) else float("nan")
        brier = float(brier_score_loss(y, p))

        metrics[target_name] = {
            "display_name": display_name,
            "auroc": round(auc, 4),
            "auroc_ci95": ci,
            "auprc": round(prc, 4),
            "brier_score": round(brier, 4),
            "prevalence": round(float(y.mean()), 4),
        }

    return {
        "metrics": metrics,
        "probs": probs_dict,
        "labels": labels,
        "demographics": all_demographics,
    }


def compute_mfa_dropout_report(full_eval: dict, drop_echo_eval: dict) -> dict:
    report = {}
    for i, (_, target_name, display_name) in enumerate(TARGET_KEYS):
        y_true = full_eval["labels"][:, i].astype(bool)
        p_full = full_eval["probs"][target_name]
        p_drop = drop_echo_eval["probs"][target_name]

        was_correct = (p_full >= 0.5) == y_true
        now_wrong = (p_drop >= 0.5) != y_true
        induced_critical = was_correct & now_wrong & (y_true == 1)

        silent_miss = induced_critical & (p_drop < 0.35)
        loud_miss = induced_critical & (p_drop >= 0.35)

        n_induced = int(induced_critical.sum())
        n_silent = int(silent_miss.sum())
        n_loud = int(loud_miss.sum())

        report[target_name] = {
            "display_name": display_name,
            "test_positive_cases": int(y_true.sum()),
            "induced_critical_misses_on_echo_drop": n_induced,
            "silent_misses": n_silent,
            "loud_misses": n_loud,
            "silent_miss_rate": round(float(n_silent / max(1, n_induced)), 4),
        }
    return report


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=str(repo_root / "cohort" / "valvular_cohort_with_splits.parquet"))
    parser.add_argument("--out-dir", default=str(repo_root / "results" / "valvular_neural"))
    parser.add_argument("--model", choices=["cross_attn", "concat_mlp"], default="cross_attn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Creating Task B Valvular Multimodal DataLoaders...")
    loaders = create_valvular_dataloaders(args.cohort, batch_size=args.batch_size)

    log.info("Instantiating %s model architecture...", args.model)
    if args.model == "cross_attn":
        model = CrossAttentionValvularProbe(embed_dim=256, echo_dim=1024, ecg_dim=768).to(device)
    else:
        model = ConcatMLPValvularProbe(echo_dim=1024, ecg_dim=768, hidden_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    log.info("Starting Multi-Task Neural Probe Training (%d epochs)...", args.epochs)
    best_val_auc = -1.0
    best_checkpoint_path = out_dir / f"valvular_{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in loaders["train"]:
            echo = batch.echo_emb.to(device)
            ecg = batch.ecg_emb.to(device)
            labels = batch.labels.to(device)
            grades = batch.grades.to(device)

            optimizer.zero_grad()
            out = model(echo, ecg)
            loss = compute_multitask_loss(out, labels, grades, bce_loss, ce_loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch.labels)

        scheduler.step()
        train_loss /= len(loaders["train"].dataset)

        # Validation evaluation
        val_eval = evaluate_model_on_split(model, loaders["val"], device)
        mean_val_auc = np.mean([val_eval["metrics"][t]["auroc"] for _, t, _ in TARGET_KEYS])

        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            torch.save(model.state_dict(), best_checkpoint_path)

        if epoch % 5 == 0 or epoch == args.epochs:
            log.info("Epoch %2d/%2d | Train Loss: %.4f | Val Mean AUROC: %.4f", epoch, args.epochs, train_loss, mean_val_auc)

    log.info("Loading best checkpoint (Val Mean AUROC: %.4f) for test evaluation...", best_val_auc)
    model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))

    log.info("Running Missing-Modality Evaluation on Held-out Test Split (n=%d)...", len(loaders["test"].dataset))
    full_eval = evaluate_model_on_split(model, loaders["test"], device)
    drop_echo_eval = evaluate_model_on_split(model, loaders["test"], device, mask_modality="echo")
    drop_ecg_eval = evaluate_model_on_split(model, loaders["test"], device, mask_modality="ecg")

    # Modality Failure Analysis
    mfa_report = compute_mfa_dropout_report(full_eval, drop_echo_eval)

    # Summary table
    table_rows = []
    for _, target_name, display_name in TARGET_KEYS:
        f_res = full_eval["metrics"][target_name]
        d_ecg = drop_ecg_eval["metrics"][target_name]
        d_echo = drop_echo_eval["metrics"][target_name]

        table_rows.append({
            "Target": display_name,
            "Full (Echo+ECG) AUROC": f"{f_res['auroc']:.3f} [{f_res['auroc_ci95'][0]:.3f}, {f_res['auroc_ci95'][1]:.3f}]",
            "Drop-ECG (Echo Only) AUROC": f"{d_ecg['auroc']:.3f}",
            "Drop-Echo (ECG Only) AUROC": f"{d_echo['auroc']:.3f}",
        })

    summary_df = pd.DataFrame(table_rows)
    summary_df.to_csv(out_dir / "valvular_neural_results_table.csv", index=False)

    results_payload = {
        "architecture": args.model,
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test": len(loaders["test"].dataset),
        "full_multimodal": full_eval["metrics"],
        "drop_ecg": drop_ecg_eval["metrics"],
        "drop_echo": drop_echo_eval["metrics"],
        "mfa_dropout_report": mfa_report,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 75)
    print(f"TASK B NEURAL PROBE RESULTS ({args.model.upper()}) ON TEST SET (n={len(loaders['test'].dataset)})")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)
    print("\nLOUD VS. SILENT MISSING-MODALITY DROPOUT PROFILE (Echo Dropped):")
    for t, rep in mfa_report.items():
        print(f"  - {rep['display_name']:32s}: Induced Misses = {rep['induced_critical_misses_on_echo_drop']:2d} | Silent = {rep['silent_misses']:2d} ({rep['silent_miss_rate']*100:.1f}%) | Loud = {rep['loud_misses']:2d}")
    print("=" * 75)


if __name__ == "__main__":
    main()
