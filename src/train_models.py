"""train_models.py — Entraînement contrôlé de tous les modèles, split fit/calib partagé.

Objectif : maîtriser θ sans fuite. On entraîne sur la SEULE portion `fit` du train
(`segment_train`), pour que la portion `calib` reste vierge — c'est sur `calib` que le
notebook calibre θ, et sur le holdout (`segment_eval`) qu'il teste. Le split est par
alerte, reproductible, et **partagé** avec le notebook via `make_split` : entraînement
et calibration voient donc exactement la même frontière fit/calib.

On (ré)entraîne tous les modèles de `eval_inference.SEQ_REGISTRY` (architectures et noms
de checkpoints = source de vérité unique) plus les tabulaires (BNN MC Dropout, XGBoost
AFT). Multi-GPU : variantes distribuées sur les `cuda:i` via une file partagée, un
thread worker par device (cf. `train_parallel`).
"""

from __future__ import annotations

import importlib
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import eval_inference as ei

MODELS_DIR = ei.MODELS_DIR
SEED = 42
CALIB_FRAC = 0.2  # fraction du train réservée à la calibration de θ (jamais entraînée)

# Recette d'entraînement par classe de modèle : (module, trainer, kwargs trainer, n_epochs).
# Les kwargs *d'architecture* viennent de `eval_inference.SEQ_REGISTRY` (source unique),
# pour garantir que les checkpoints se rechargent à l'inférence.
TRAINER_BY_CLS = {
    "NeuralHawkesGRUv2": ("neural_hawkes_v2", "NeuralHawkesTrainer", dict(lr=5e-4, reg_weight=1.0), 80),
    "NeuralHawkesTransformer": ("neural_hawkes_v2", "NeuralHawkesTrainer", dict(lr=5e-4, reg_weight=1.0), 80),
    "GaussianHawkesGRU": ("neural_hawkes_v3", "GaussianHawkesTrainer", dict(lr=5e-4, reg_weight=0.5), 80),
    "GaussianHawkesTransformer": ("neural_hawkes_v3", "GaussianHawkesTrainer", dict(lr=5e-4, reg_weight=0.5), 80),
    "MCDropoutHawkesTransformer": ("bayesian_hawkes", "BayesianHawkesTrainer", dict(lr=5e-4, reg_weight=1.0), 60),
    "VariationalHawkesTransformer": ("bayesian_hawkes", "VariationalHawkesTrainer", dict(lr=5e-4, kl_weight=1e-4, reg_weight=1.0), 60),
    "SpatialMoETransformer": ("spatial_moe_model", "SpatialMoETrainer", dict(lr=5e-4, reg_weight=0.5, balance_weight=0.1), 80),
}


# ─── Split partagé fit / calibration ──────────────────────────────────────────

def make_split(
    train_csv: str | Path, seed: int = SEED, calib_frac: float = CALIB_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare le train PAR ALERTE en (fit, calib). Déterministe — à partager tel quel.

    `fit` sert à entraîner les modèles ; `calib` (jamais entraînée) sert à calibrer θ.
    `load_eval_alerts` trie les éclairs de façon déterministe, donc le split est
    reproductible d'un appel à l'autre (entraînement ⇄ notebook).

    Returns:
        (fit_alerts, calib_alerts) — DataFrames d'éclairs en alerte.
    """
    alerts = ei.load_eval_alerts(train_csv)
    groups = (alerts["airport"].astype(str) + "_" + alerts["airport_alert_id"].astype(str)).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
    fit_idx, calib_idx = next(splitter.split(alerts, groups=groups))
    return alerts.iloc[fit_idx].copy(), alerts.iloc[calib_idx].copy()


# ─── Entraînement des modèles séquentiels (multi-GPU) ─────────────────────────

def _instantiate(spec: dict):
    """Construit un modèle (non entraîné) depuis son entrée de registre."""
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["cls"])(**spec["kwargs"])


def train_sequential(
    fit_alerts: pd.DataFrame,
    models_dir: Path,
    devices: list[torch.device],
    epochs_scale: float = 1.0,
    batch_size: int = 32,
    verbose: bool = True,
) -> list[str]:
    """Entraîne tous les modèles séquentiels du registre sur `fit_alerts`, sauve les .pt.

    Les sessions sont préparées une fois par type de features (12 / 20). Les variantes
    sont distribuées sur les devices via une file partagée (un thread par GPU). Chaque
    modèle est sauvé sous le nom attendu par `eval_inference.SEQ_REGISTRY`.

    Returns:
        Liste des chemins de checkpoints écrits.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    specs = ei.SEQ_REGISTRY
    needed = {s["prepare"] for s in specs}

    sessions_by_kind: dict[str, list[dict]] = {}
    if "v2" in needed:
        import neural_hawkes_v2 as nh2

        print("[train] Préparation des sessions (12 features)…")
        sessions_by_kind["v2"] = nh2.prepare_sessions_v2(fit_alerts)
    if "spatial" in needed:
        from spatial_features import prepare_sessions_spatial

        print("[train] Préparation des sessions spatiales (20 features)…")
        sessions_by_kind["spatial"] = prepare_sessions_spatial(fit_alerts)

    task_q: queue.Queue = queue.Queue()
    for spec in specs:
        task_q.put(spec)
    saved: list[str] = []
    print_lock = threading.Lock()

    def worker(device: torch.device) -> None:
        while True:
            try:
                spec = task_q.get_nowait()
            except queue.Empty:
                return
            try:
                module, trainer_name, trainer_kwargs, base_epochs = TRAINER_BY_CLS[spec["cls"]]
                n_epochs = max(1, round(base_epochs * epochs_scale))
                with print_lock:
                    print(f"[train][{device}] {spec['name']} — {n_epochs} epochs…")
                model = _instantiate(spec)
                trainer_cls = getattr(importlib.import_module(module), trainer_name)
                trainer = trainer_cls(model, device=device, **trainer_kwargs)
                trainer.fit(
                    sessions_by_kind[spec["prepare"]],
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                )
                path = models_dir / spec["file"]
                torch.save(model.state_dict(), str(path))
                with print_lock:
                    saved.append(str(path))
                    print(f"[train][{device}] ✓ {spec['name']} → {path.name}")
                del model, trainer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                task_q.task_done()

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return saved


# ─── Entraînement des modèles tabulaires (BNN, XGBoost) ───────────────────────

def _tabular_features(alerts: pd.DataFrame) -> pd.DataFrame:
    """Features tabulaires de survie (build_features), `icloud` en entier."""
    import features as feat_module

    feat_df = feat_module.build_features(alerts)
    feat_df["icloud"] = feat_df["icloud"].astype(int)
    feat_df["airport"] = feat_df["airport"].astype(str)
    feat_df["airport_alert_id"] = feat_df["airport_alert_id"].astype(str)
    return feat_df


def _split_by_alert(feat_df: pd.DataFrame, frac: float, seed: int):
    """Sépare un DataFrame de features par alerte (pour la validation interne au fit)."""
    groups = (feat_df["airport"] + "_" + feat_df["airport_alert_id"]).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    tr_idx, va_idx = next(splitter.split(feat_df, groups=groups))
    return feat_df.iloc[tr_idx].copy(), feat_df.iloc[va_idx].copy()


def _train_bnn(tab_train, tab_val, models_dir: Path, device: torch.device,
               n_epochs: int = 80, batch_size: int = 512, lr: float = 1e-3) -> str:
    """Entraîne le BNN MC Dropout sur `tab_train` et sauve checkpoint + scaler."""
    from bnn_model import FEATURE_COLS, TARGET_COL, BayesianMLP

    scaler = StandardScaler()
    x_train = scaler.fit_transform(tab_train[FEATURE_COLS].astype(float).values)
    y_train = tab_train[TARGET_COL].values.astype(np.float32)

    x_t = torch.FloatTensor(x_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    model = BayesianMLP(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=20.0)

    model.train()
    n = x_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = criterion(model(x_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    path = models_dir / "bnn_mc_dropout.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": FEATURE_COLS,
    }, str(path))
    return str(path)


def _train_xgboost(tab_train, tab_val, models_dir: Path) -> str:
    """Entraîne le XGBoost survival:aft et sauve `xgboost_aft.json`."""
    import evaluation

    model, _ = evaluation.train_xgboost_survival(tab_train, tab_val)
    path = models_dir / "xgboost_aft.json"
    model.save_model(str(path))
    return str(path)


def train_tabular(
    fit_alerts: pd.DataFrame,
    models_dir: Path,
    device: torch.device,
    seed: int = SEED,
    val_frac: float = 0.15,
) -> list[str]:
    """Entraîne BNN + XGBoost sur `fit_alerts` (validation interne au fit, calib intacte)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    feat_df = _tabular_features(fit_alerts)
    # Validation carvée DANS le fit : XGBoost fait de l'early-stopping dessus → ne jamais
    # y mettre le set de calibration de θ.
    tab_train, tab_val = _split_by_alert(feat_df, val_frac, seed)

    saved = []
    print("[train] BNN MC Dropout…")
    saved.append(_train_bnn(tab_train, tab_val, models_dir, device))
    print("[train] XGBoost survival:aft…")
    saved.append(_train_xgboost(tab_train, tab_val, models_dir))
    return saved


# ─── Orchestration ────────────────────────────────────────────────────────────

def train_all(
    train_csv: str | Path,
    models_dir: Path | None = None,
    seed: int = SEED,
    calib_frac: float = CALIB_FRAC,
    devices: list[torch.device] | None = None,
    epochs_scale: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Split fit/calib, entraîne tous les modèles sur `fit`, sauve les checkpoints.

    `calib` n'est PAS utilisée ici (elle reste vierge pour la calibration de θ par le
    notebook, via le même `make_split`).

    Returns:
        Dict {fit_alerts, calib_alerts, saved} pour réutilisation par le notebook.
    """
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    devices = devices or ei.detect_devices()
    fit_alerts, calib_alerts = make_split(train_csv, seed, calib_frac)

    def _n(df):
        return df.groupby(["airport", "airport_alert_id"]).ngroups

    print(f"[train] Devices : {[str(d) for d in devices]}")
    print(f"[train] fit : {_n(fit_alerts)} alertes / {len(fit_alerts)} éclairs")
    print(f"[train] calib (θ, vierge) : {_n(calib_alerts)} alertes / {len(calib_alerts)} éclairs")

    saved = train_sequential(fit_alerts, models_dir, devices, epochs_scale, verbose=verbose)
    saved += train_tabular(fit_alerts, models_dir, devices[0], seed)
    print(f"[train] {len(saved)} checkpoints écrits dans {models_dir}")
    return {"fit_alerts": fit_alerts, "calib_alerts": calib_alerts, "saved": saved}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement contrôlé fit/calib.")
    parser.add_argument(
        "--train-csv",
        default="/home/vm2lucas/Bureau/Hackathon/hackathon2026/data/segment_alerts_all_airports_train.csv",
    )
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--calib-frac", type=float, default=CALIB_FRAC)
    parser.add_argument("--epochs-scale", type=float, default=1.0)
    args = parser.parse_args()

    train_all(
        args.train_csv,
        models_dir=Path(args.models_dir),
        seed=args.seed,
        calib_frac=args.calib_frac,
        epochs_scale=args.epochs_scale,
    )
