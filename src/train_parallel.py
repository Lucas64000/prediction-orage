"""Entraînement + évaluation des modèles Hawkes V3 répartis sur plusieurs GPU.

Détecte les GPU disponibles et lance les variantes (V3a GRU, V3b/V3c Transformer,
V3d par aéroport) en parallèle — une par GPU — via une file de travail partagée.
Pas de scatter/gather : chaque entraînement est indépendant sur son propre GPU.

Les fonctions des modèles sont appelées via leur module d'origine
(`neural_hawkes_v3.GaussianHawkesTrainer`, `neural_hawkes_v2.prepare_sessions_v2`,
`data_loader.load_raw`, …) pour garder visible d'où vient chaque symbole.

Usage notebook distant :
    import train_parallel
    train_parallel.run_v3(data_csv=".../train.csv", output_dir="outputs")

Usage CLI local :
    python src/train_parallel.py --data-csv .../train.csv --output-dir outputs
"""

import os
import sys
import json
import time
import queue
import argparse
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import data_loader
import neural_hawkes_v2
import neural_hawkes_v3


# =============================================================================
# Détection GPU
# =============================================================================

def detect_devices():
    """Liste des devices à utiliser : tous les GPU CUDA, sinon le CPU."""
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


# =============================================================================
# Dispatcher multi-GPU
# =============================================================================

# Sérialise les prints d'évaluation pour que les sorties des threads ne
# s'entrelacent pas dans la même cellule.
_print_lock = threading.Lock()


def _run_variant(variant, device, model_dir):
    """Entraîne puis évalue une variante sur un device donné."""
    name = variant["name"]
    tag = f"{name} [{device}]"

    model = variant["build"]().to(device)
    trainer = neural_hawkes_v3.GaussianHawkesTrainer(
        model,
        lr=variant.get("lr", 5e-4),
        reg_weight=variant.get("reg_weight", 0.5),
        device=device,
    )
    trainer.fit(
        variant["train"],
        n_epochs=variant.get("n_epochs", 80),
        batch_size=variant.get("batch_size", 32),
        desc=tag,
    )

    # Libère le cache fragmenté du training avant l'éval MC qui réclame de gros
    # blocs contigus (sinon OOM par fragmentation sur les gros Transformers).
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with _print_lock:
        r = neural_hawkes_v3.evaluate_gaussian_model(
            trainer,
            variant["test"],
            label=name,
            with_uncertainty=variant.get("with_uncertainty", False),
            n_mc=variant.get("n_mc", 50),
            mc_batch_size=variant.get("mc_batch_size", 10),
            max_seq_len=variant.get("max_seq_len", 200),
        )

    save_name = variant.get("save_name")
    if save_name:
        torch.save(model.state_dict(), str(model_dir / save_name))

    # Libère la mémoire avant la prochaine variante sur ce GPU.
    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return name, r


def run_variants_parallel(variants, devices, model_dir):
    """Dispatch via file partagée + 1 thread par device. Renvoie {name: result}."""
    work_queue = queue.Queue()
    for v in variants:
        work_queue.put(v)

    results = {}
    results_lock = threading.Lock()
    errors = []

    def worker(device):
        while True:
            try:
                v = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                name, r = _run_variant(v, device, model_dir)
                if r is not None:
                    with results_lock:
                        results[name] = r
            except Exception as exc:  # noqa: BLE001
                with results_lock:
                    errors.append((v.get("name", "?"), str(device), repr(exc)))
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(d,), name=f"gpu-worker-{d}")
        for d in devices
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print("\n❌ Échecs :")
        for name, dev, err in errors:
            print(f"  {name} [{dev}] → {err}")

    return results


# =============================================================================
# Définition des variantes
# =============================================================================

def _build_wave1(train_sessions, test_sessions, n_epochs):
    """V3a (GRU), V3b (Transformer), V3c (Transformer Large) — modèles globaux."""
    return [
        {
            "name": "V3a GRU",
            "build": lambda: neural_hawkes_v3.GaussianHawkesGRU(
                input_dim=12, hidden_dim=64, dropout=0.15
            ),
            "train": train_sessions,
            "test": test_sessions,
            "lr": 5e-4,
            "n_epochs": n_epochs,
            "batch_size": 32,
            "with_uncertainty": True,
            "save_name": "hawkes_gru_gaussian.pt",
        },
        {
            "name": "V3b TF",
            "build": lambda: neural_hawkes_v3.GaussianHawkesTransformer(
                input_dim=12, d_model=64, nhead=4, num_layers=3,
                dim_feedforward=128, dropout=0.15,
            ),
            "train": train_sessions,
            "test": test_sessions,
            "lr": 5e-4,
            "n_epochs": n_epochs,
            "batch_size": 32,
            "with_uncertainty": True,
            "save_name": "hawkes_tf_gaussian.pt",
        },
        {
            "name": "V3c TF-Large",
            "build": lambda: neural_hawkes_v3.GaussianHawkesTransformer(
                input_dim=12, d_model=128, nhead=8, num_layers=4,
                dim_feedforward=256, dropout=0.15,
            ),
            "train": train_sessions,
            "test": test_sessions,
            "lr": 3e-4,
            "n_epochs": n_epochs,
            "batch_size": 32,
            "with_uncertainty": True,
            "save_name": "hawkes_tf_large_gaussian.pt",
        },
    ]


def _build_wave2(train_sessions, test_sessions, airports, n_epochs):
    """V3d — un Transformer par aéroport (skip si trop peu de sessions)."""
    wave = []
    for ap in airports:
        ap_train = [s for s in train_sessions if s["airport"] == ap]
        ap_test = [s for s in test_sessions if s["airport"] == ap]
        if len(ap_train) < 10 or len(ap_test) < 3:
            print(
                f"  ⚠ {ap} : trop peu de sessions "
                f"({len(ap_train)} train / {len(ap_test)} test), variante ignorée"
            )
            continue
        wave.append(
            {
                "name": f"V3d {ap}",
                "build": lambda: neural_hawkes_v3.GaussianHawkesTransformer(
                    input_dim=12, d_model=64, nhead=4, num_layers=2,
                    dim_feedforward=128, dropout=0.15,
                ),
                "train": ap_train,
                "test": ap_test,
                "lr": 5e-4,
                "n_epochs": n_epochs,
                "batch_size": min(16, len(ap_train)),
                "with_uncertainty": False,
                "save_name": f"hawkes_gaussian_{ap.lower()}.pt",
            }
        )
    return wave


# =============================================================================
# Pipeline complète
# =============================================================================

def run_v3(
    data_csv,
    output_dir,
    test_ratio=0.2,
    n_epochs=80,
    devices=None,
    seed=42,
):
    """Pipeline V3 complète sur N GPU. Renvoie le dict des résultats par variante."""
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    devices = devices if devices is not None else detect_devices()

    t0 = time.perf_counter()
    print(f"Devices : {[str(d) for d in devices]}")

    print("\nLoading data...")
    df = data_loader.load_raw(data_csv)
    alerts = data_loader.load_alerts(df)

    print("Preparing sessions...")
    sessions = neural_hawkes_v2.prepare_sessions_v2(alerts)
    print(f"  {len(sessions)} sessions")

    np.random.seed(seed)
    idx = np.random.permutation(len(sessions))
    split = int(len(sessions) * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]
    airports = sorted({s["airport"] for s in sessions})
    print(f"  Train: {len(train_sessions)} | Test: {len(test_sessions)}")

    # Vague 1 : modèles globaux
    print("\n" + "=" * 60)
    print(f"  Vague 1 — modèles globaux sur {len(devices)} device(s)")
    print("=" * 60)
    wave1 = _build_wave1(train_sessions, test_sessions, n_epochs)
    results = run_variants_parallel(wave1, devices, model_dir)

    # Vague 2 : un modèle par aéroport
    print("\n" + "=" * 60)
    print(f"  Vague 2 — par aéroport sur {len(devices)} device(s)")
    print("=" * 60)
    wave2 = _build_wave2(train_sessions, test_sessions, airports, n_epochs)
    results.update(run_variants_parallel(wave2, devices, model_dir))

    _aggregate_v3d(results, airports)
    _print_summary(results)
    _persist(results, output_dir)

    elapsed = time.perf_counter() - t0
    print(f"\nRésultats sauvegardés dans {output_dir}")
    print(f"Temps total : {int(elapsed // 60):02d}m {elapsed % 60:05.2f}s — Terminé !")
    return results


def _aggregate_v3d(results, airports):
    """Agrège les erreurs des modèles par aéroport en un ensemble V3d."""
    import pandas as pd

    per_ap = [
        results[f"V3d {ap}"]["errors_df"]
        for ap in airports
        if f"V3d {ap}" in results and results[f"V3d {ap}"]["errors_df"] is not None
    ]
    if not per_ap:
        return
    combined = pd.concat(per_ap, ignore_index=True)
    mae_c = np.mean(np.abs(combined["true"] - combined["pred"]))
    rmse_c = np.sqrt(np.mean((combined["true"] - combined["pred"]) ** 2))
    bias_c = np.mean(combined["pred"] - combined["true"])
    med_c = np.median(np.abs(combined["true"] - combined["pred"]))
    p90_c = np.percentile(np.abs(combined["true"] - combined["pred"]), 90)
    print(f"\n  Ensemble V3d → MAE: {mae_c:.2f} | Biais: {bias_c:+.2f}")
    results["V3d Ensemble"] = {
        "label": "Ensemble par aéroport (Gaussian)",
        "mae": mae_c, "rmse": rmse_c, "median": med_c,
        "bias": bias_c, "p90": p90_c, "errors_df": combined,
    }


def _print_summary(results):
    """Tableau récapitulatif des variantes principales."""
    print("\n\n" + "=" * 75)
    print("  RÉSUMÉ FINAL PHASE 2bis (multi-GPU)")
    print("=" * 75)
    print(
        f"  {'Variante':40s} | {'MAE':>6s} | {'RMSE':>6s} | {'Méd.':>6s} | "
        f"{'Biais':>7s} | {'P90':>6s} | {'σ':>5s} | {'1σ':>5s}"
    )
    print("-" * 75)
    print(
        f"  {'[Réf] Baseline 30 min':40s} | {'60.26':>6s} | {'---':>6s} | "
        f"{'---':>6s} | {'-52.32':>7s} | {'---':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print("-" * 75)
    for key in ["V3a GRU", "V3b TF", "V3c TF-Large", "V3d Ensemble"]:
        if key not in results:
            continue
        r = results[key]
        unc = f"{r.get('mean_uncertainty', 0):.1f}" if "mean_uncertainty" in r else "---"
        cal1 = (
            f"{r.get('calibration_1std', 0) * 100:.0f}%"
            if "calibration_1std" in r
            else "---"
        )
        print(
            f"  {key:40s} | {r['mae']:6.2f} | {r.get('rmse', 0):6.2f} | "
            f"{r.get('median', 0):6.2f} | {r['bias']:+7.2f} | "
            f"{r.get('p90', 0):6.2f} | {unc:>5s} | {cal1:>5s}"
        )
    print("=" * 75)


def _persist(results, output_dir):
    """Écrit les DataFrames d'erreurs (.parquet) et le résumé (.json)."""
    for key, r in results.items():
        if "errors_df" in r and r["errors_df"] is not None:
            out = output_dir / f"errors_v3_{key.replace(' ', '_').lower()}.parquet"
            r["errors_df"].to_parquet(out, index=False)
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("errors_df", "airport_metrics")}
        for k, v in results.items()
    }
    with open(output_dir / "phase2bis_v3_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-csv", required=True, help="CSV d'entraînement segmenté.")
    parser.add_argument("--output-dir", default="outputs", help="Dossier de sortie.")
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_v3(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
