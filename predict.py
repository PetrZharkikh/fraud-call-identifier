#!/usr/bin/env python3
"""
Классификация WAV: Kairos ASR, сценарный скоринг, CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_):
        return it

from scenario_fraud import label_from_scenario, scenario_score_breakdown


def iter_wav_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                yield os.path.join(dirpath, name)


def main() -> int:
    if sys.version_info < (3, 10):
        print("Нужен Python 3.10+. Пример: python3 predict.py …", file=sys.stderr)
        return 2
    try:
        from asr_kairos import transcribe_wav
    except ImportError as e:
        print("Установите зависимости: pip install -r requirements.txt\n", e, file=sys.stderr)
        return 2

    p = argparse.ArgumentParser(
        description="Vishing: папка с WAV, транскрипт (Kairos), метки в CSV",
        epilog="Порог по умолчанию 5.5 (подобран на калибровочном наборе); см. calibrate.py",
    )
    p.add_argument(
        "input_dir",
        help="Каталог с .wav (обход рекурсивный)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="results.csv",
        help="Выходной CSV (по умолчанию results.csv)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=5.5,
        help="Порог fraud_score: при total ≥ порога класс 0 (мошенничество)",
    )
    p.add_argument("--with-scores", action="store_true", help="Добавить колонку fraud_score")
    p.add_argument("--verbose", action="store_true", help="Подробный лог в stderr")
    p.add_argument(
        "--verbose-chars",
        type=int,
        default=200,
        metavar="N",
        help="Длина фрагмента текста в логе (0 = весь текст)",
    )
    p.add_argument("--basename-only", action="store_true", help="В CSV только имя файла, без путей")
    p.add_argument("--kairos-device", default=os.environ.get("KAIROS_DEVICE", "auto"))
    p.add_argument("--kairos-model-path", default=None)
    p.add_argument("--kairos-force-download", action="store_true")
    p.add_argument("--kairos-pause-threshold", type=float, default=2.0)
    args = p.parse_args()

    root = os.path.abspath(args.input_dir)
    if not os.path.isdir(root):
        print(f"Не каталог: {root}", file=sys.stderr)
        return 2

    files = sorted(iter_wav_files(root))
    if not files:
        print(f"Нет .wav в {root}", file=sys.stderr)
        return 1

    rows: list[tuple[str, int, float, str]] = []
    for path in tqdm(files, desc="Файлы"):
        rel = os.path.relpath(path, root)
        name_key = os.path.basename(path) if args.basename_only else rel.replace(os.sep, "/")
        try:
            text = transcribe_wav(
                path,
                device=args.kairos_device,
                model_path=args.kairos_model_path,
                force_download=args.kairos_force_download,
                pause_threshold=args.kairos_pause_threshold,
            )
        except Exception as e:
            print(f"{path}: ASR: {e}", file=sys.stderr)
            text = ""

        sb = scenario_score_breakdown(text)
        sc = sb.total
        lab = label_from_scenario(sc, args.threshold)
        rows.append((name_key, lab, sc, text))

        if args.verbose:
            n = args.verbose_chars
            snippet = text if n <= 0 else text[:n]
            if n > 0 and len(text) > n:
                snippet += "…"
            active = [k for k, v in sb.categories.items() if v]
            leg = f"legit={sb.legit_adj:+.1f}" if sb.legit_adj else "legit=0"
            if sb.legit_tags:
                leg += f"({','.join(sb.legit_tags)})"
            reg = f"reg={sb.regulation_bonus:.1f}"
            if sb.regulation_tags:
                reg += f"({','.join(sb.regulation_tags[:4])}{'…' if len(sb.regulation_tags) > 4 else ''})"
            print(
                f"{name_key}\ttotal={sc:.2f} cat={sb.category_sum:.1f} combo={sb.combo_sum:.1f} "
                f"chain={'Y' if sb.chain_ok else 'N'}({sb.chain_kind}+{sb.chain_bonus:.0f})\t{leg}\t{reg}\t"
                f"phr={sb.n_phrases}\thit={','.join(active) or '-'}\tlabel={lab}\t{snippet!r}",
                file=sys.stderr,
            )

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = ["Название файла", "label"]
    if args.with_scores:
        header.append("fraud_score")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for name_key, lab, sc, _text in rows:
            if args.with_scores:
                w.writerow([name_key, lab, f"{sc:.4f}"])
            else:
                w.writerow([name_key, lab])

    print(f"Готово: {out_path} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
