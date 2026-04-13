#!/usr/bin/env python3
"""
Калибровка порога на размеченных данных: root/Fraud и root/NotFraud.

Kairos + scenario_fraud, поиск порога с max F1 (класс «мошенничество»).

  python3 calibrate.py path/to/labeled_root
  python3 calibrate.py path/to/labeled_root --dump 200
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main() -> int:
    if sys.version_info < (3, 10):
        print("Нужен Python 3.10+", file=sys.stderr)
        return 2
    try:
        from asr_kairos import transcribe_wav
    except ImportError as e:
        print("pip install -r requirements.txt", e, file=sys.stderr)
        return 2

    from scenario_fraud import label_from_scenario, scenario_score_breakdown

    p = argparse.ArgumentParser(description="Калибровка порога (размеченные Fraud / NotFraud)")
    p.add_argument("root", help="Корень с подпапками Fraud и NotFraud")
    p.add_argument("--threshold-min", type=float, default=4.0)
    p.add_argument("--threshold-max", type=float, default=18.0)
    p.add_argument("--threshold-step", type=float, default=0.5)
    p.add_argument(
        "--reference-thr",
        type=float,
        default=5.5,
        help="Матрица ошибок при этом пороге (совместить с predict.py --threshold)",
    )
    p.add_argument("--dump", type=int, default=0, metavar="N", help="Длина фрагмента текста у ошибок")
    p.add_argument("--kairos-device", default=os.environ.get("KAIROS_DEVICE", "auto"))
    p.add_argument("--kairos-model-path", default=None)
    p.add_argument("--kairos-pause-threshold", type=float, default=2.0)
    args = p.parse_args()

    root = os.path.abspath(args.root)
    fraud_dir = os.path.join(root, "Fraud")
    legit_dir = os.path.join(root, "NotFraud")
    if not os.path.isdir(fraud_dir) or not os.path.isdir(legit_dir):
        print(f"Ожидаются каталоги {fraud_dir} и {legit_dir}", file=sys.stderr)
        return 2

    records: list[tuple[str, int, str, object]] = []

    def walk(d: str, y: int):
        for dirpath, _, files in os.walk(d):
            for name in sorted(files):
                if not name.lower().endswith(".wav"):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, root)
                try:
                    text = transcribe_wav(
                        path,
                        device=args.kairos_device,
                        model_path=args.kairos_model_path,
                        pause_threshold=args.kairos_pause_threshold,
                    )
                except Exception as e:
                    print(f"skip {rel}: {e}", file=sys.stderr)
                    continue
                records.append((rel, y, text, scenario_score_breakdown(text)))

    walk(fraud_dir, 1)
    walk(legit_dir, 0)
    if not records:
        print("Нет wav", file=sys.stderr)
        return 1

    def predict_fraud(bd, thr: float) -> int:
        return 1 if label_from_scenario(bd.total, thr) == 0 else 0

    best_t = args.threshold_min
    best_f1 = -1.0
    best_acc = 0.0
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        tp = fp = tn = fn = 0
        for _rel, y, _txt, bd in records:
            pred = predict_fraud(bd, t)
            if y == 1 and pred == 1:
                tp += 1
            elif y == 1 and pred == 0:
                fn += 1
            elif y == 0 and pred == 1:
                fp += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(records)
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_t = t
        t += args.threshold_step

    print(
        f"n={len(records)} | лучший порог (max F1 fraud): thr={best_t:.2f} "
        f"acc={best_acc:.3f} F1(fraud)={best_f1:.3f}"
    )

    thr0 = args.reference_thr
    tp = fp = tn = fn = 0
    errors: list[tuple[str, int, int, object, str]] = []
    for rel, y, txt, bd in records:
        pred = predict_fraud(bd, thr0)
        if y == 1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
            errors.append((rel, y, pred, bd, txt))
        elif y == 0 and pred == 1:
            fp += 1
            errors.append((rel, y, pred, bd, txt))
        else:
            tn += 1
    acc0 = (tp + tn) / len(records)
    print(f"\nПри thr={thr0}: acc={acc0:.3f} TP={tp} FN={fn} FP={fp} TN={tn}")

    if args.dump and errors:
        print("\n--- Ошибки (FN / FP) ---")
        for rel, y, pred, bd, txt in sorted(errors):
            tag = "FN" if y == 1 else "FP"
            snip = (txt[: args.dump] + "…") if len(txt) > args.dump else txt
            active = [k for k, v in bd.categories.items() if v]
            leg = f" legit={bd.legit_adj:+.1f}" if bd.legit_adj else ""
            print(
                f"{tag} {rel} | total={bd.total:.2f} cat={bd.category_sum:.1f} combo={bd.combo_sum:.1f}"
                f"{leg} chain={bd.chain_ok}\thit={','.join(active) or '-'} | {snip!r}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
