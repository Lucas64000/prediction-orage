"""
neural_hawkes_v3.py - Phase 2bis : itération finale.

Améliorations par rapport à v2 :
  - Log-scale target : log(1 + time_to_end) pour normaliser la distribution asymétrique
  - Gaussian NLL : le modèle prédit (mu, log_sigma) pour une incertitude calibrée
  - Tête de régression plus profonde avec skip connection
"""

import os
import sys
import json
import math
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Modules du projet — importés en entier pour appeler via `module.fonction()`
# et garder visible d'où vient chaque symbole.
import data_loader
import neural_hawkes_v2

# ----------------------------------------------------═══════════════════════════
# 1. DATASET avec log-target
# ----------------------------------------------------═══════════════════════════


class HawkesDatasetV3(Dataset):
    """Dataset avec time_to_end en log-scale."""

    def __init__(self, sessions, max_len=200):
        self.sessions = []
        self.max_len = max_len
        self.feat_dim = None

        for s in sessions:
            if len(s["times"]) < 3:
                continue
            n = min(len(s["times"]), max_len)
            self.sessions.append(
                {
                    "times": s["times"][:n],
                    "features": s["features"][:n],
                    "time_to_end": s["time_to_end"][:n],
                    "n": n,
                    "airport": s["airport"],
                }
            )
            if self.feat_dim is None:
                self.feat_dim = s["features"].shape[1]

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        s = self.sessions[idx]
        n = s["n"]

        times = np.zeros(self.max_len)
        features = np.zeros((self.max_len, self.feat_dim))
        time_to_end = np.zeros(self.max_len)
        log_tte = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)

        times[:n] = s["times"]
        features[:n] = s["features"]
        time_to_end[:n] = s["time_to_end"]
        log_tte[:n] = np.log1p(s["time_to_end"])  # log(1 + t)
        mask[:n] = 1.0

        dt = np.zeros(self.max_len)
        dt[1:n] = np.diff(times[:n])

        padding_mask = np.ones(self.max_len, dtype=bool)
        padding_mask[:n] = False

        return {
            "times": torch.FloatTensor(times),
            "features": torch.FloatTensor(features),
            "delta_times": torch.FloatTensor(dt).unsqueeze(-1),
            "time_to_end": torch.FloatTensor(time_to_end),
            "log_tte": torch.FloatTensor(log_tte),
            "mask": torch.FloatTensor(mask),
            "padding_mask": torch.BoolTensor(padding_mask),
            "n": n,
        }


# ----------------------------------------------------═══════════════════════════
# 2. MODÈLES avec Gaussian NLL (prédit mu + sigma)
# ----------------------------------------------------═══════════════════════════


class GaussianHawkesTransformer(nn.Module):
    """
    Transformer Hawkes avec sortie gaussienne.
    Prédit (mu, log_sigma) en log-space, donc :
      - pred_minutes = exp(mu) - 1
      - incertitude calculée via la distribution en log-space
    """

    def __init__(
        self,
        input_dim=12,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.15,
    ):
        super().__init__()
        self.d_model = d_model

        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = neural_hawkes_v2.ContinuousPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Tête intensité (inchangée)
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.softplus = nn.Softplus()

        # Tête gaussienne : prédit mu et log_sigma en log-space
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(d_model // 2, 1)  # mu en log-space
        self.log_sigma_head = nn.Linear(d_model // 2, 1)  # log(sigma)

        self.dropout_input = nn.Dropout(dropout)

    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        B, S, _ = event_features.shape

        x = self.feature_proj(event_features)
        pe = self.pos_encoder(timestamps)
        x = self.dropout_input(x + pe)

        causal_mask = self.generate_causal_mask(S).to(x.device)
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Intensité
        combined = torch.cat([h, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_head(combined)).squeeze(-1)

        # Régression gaussienne en log-space
        reg_h = self.regression_head(h)
        mu = self.mu_head(reg_h).squeeze(-1)  # mu en log-space
        log_sigma = self.log_sigma_head(reg_h).squeeze(-1)  # log(sigma)
        log_sigma = torch.clamp(log_sigma, min=-4, max=3)  # stabilité numérique

        return intensity, mu, log_sigma, h


class GaussianHawkesGRU(nn.Module):
    """GRU avec sortie gaussienne en log-space."""

    def __init__(self, input_dim=12, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.softplus = nn.Softplus()

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.log_sigma_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        gru_out, _ = self.gru(event_features)
        combined = torch.cat([gru_out, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_layer(combined)).squeeze(-1)

        reg_h = self.regression_head(gru_out)
        mu = self.mu_head(reg_h).squeeze(-1)
        log_sigma = torch.clamp(self.log_sigma_head(reg_h).squeeze(-1), min=-4, max=3)
        return intensity, mu, log_sigma, gru_out


# ----------------------------------------------------═══════════════════════════
# 3. TRAINER avec Gaussian NLL
# ----------------------------------------------------═══════════════════════════


def _truncate_window(session_features, session_times, max_seq_len):
    """Garde les `max_seq_len` derniers événements et shifte les times à t=0.

    Le shift à 0 est important : ContinuousPositionalEncoding est calé sur les
    valeurs de timestamps. À l'entraînement, HawkesDatasetV3 tronque aux 200
    *premiers* événements (times démarrant à 0). À l'éval on garde les
    *derniers* (plus informatifs pour prédire la fin) mais on les remet à 0
    pour matcher la distribution de PE vue à l'entraînement.
    """
    n = len(session_times)
    if n <= max_seq_len:
        return session_features, session_times
    features = session_features[-max_seq_len:]
    times = session_times[-max_seq_len:] - session_times[-max_seq_len]
    return features, times


class GaussianHawkesTrainer:
    """Trainer avec NLL processus ponctuel + Gaussian NLL en log-space."""

    def __init__(self, model, lr=5e-4, reg_weight=0.5, device=None):
        device = device if device is not None else neural_hawkes_v2.DEVICE
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=80, eta_min=1e-6
        )
        self.device = device
        self.reg_weight = reg_weight

    def gaussian_nll(self, mu, log_sigma, target, mask):
        """
        NLL gaussienne : -log N(target | mu, sigma²)
        = 0.5 * [(target - mu)² / sigma² + 2*log_sigma + log(2π)]
        """
        sigma = torch.exp(log_sigma)
        nll = 0.5 * (
            (target - mu) ** 2 / (sigma**2 + 1e-8)
            + 2 * log_sigma
            + math.log(2 * math.pi)
        )
        return (nll * mask).sum() / (mask.sum() + 1e-8)

    def combined_loss(self, intensity, mu, log_sigma, delta_times, log_tte, mask):
        # NLL processus ponctuel
        log_lambda = torch.log(intensity + 1e-8) * mask
        compensator = intensity * delta_times.squeeze(-1) * mask
        pp_nll = (-log_lambda.sum() + compensator.sum()) / (mask.sum() + 1e-8)

        # Gaussian NLL en log-space
        g_nll = self.gaussian_nll(mu, log_sigma, log_tte, mask)

        return pp_nll + self.reg_weight * g_nll, pp_nll.item(), g_nll.item()

    def fit(
        self, train_sessions, n_epochs=80, batch_size=32, verbose=True, desc="Train"
    ):
        dataset = HawkesDatasetV3(train_sessions)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        epoch_bar = tqdm(range(n_epochs), desc=desc, disable=not verbose, leave=True)
        for epoch in epoch_bar:
            total_pp = total_g = 0
            n_batches = 0

            for batch in loader:
                features = batch["features"].to(self.device)
                times = batch["times"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                log_tte = batch["log_tte"].to(self.device)
                mask = batch["mask"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                intensity, mu, log_sigma, _ = self.model(
                    features, times, delta_times, padding_mask
                )
                loss, pp_val, g_val = self.combined_loss(
                    intensity, mu, log_sigma, delta_times, log_tte, mask
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_pp += pp_val
                total_g += g_val
                n_batches += 1

            avg_pp = total_pp / n_batches
            avg_g = total_g / n_batches
            self.scheduler.step()

            if avg_g < best_loss - 1e-4:
                best_loss = avg_g
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            epoch_bar.set_postfix(
                pp=f"{avg_pp:.4f}",
                g=f"{avg_g:.4f}",
                best=f"{best_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.1e}",
                pat=patience_counter,
            )

            if patience_counter >= 20:
                epoch_bar.write(f"  Early stopping at epoch {epoch + 1}")
                epoch_bar.close()
                break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict_time_remaining(self, session_features, session_times, max_seq_len=200):
        """Prédiction ponctuelle : exp(mu) - 1.

        max_seq_len : tronque aux derniers événements pour matcher la longueur
        vue à l'entraînement (HawkesDatasetV3 tronque à 200) et éviter l'OOM
        sur les très longues sessions.
        """
        self.model.eval()
        with torch.no_grad():
            session_features, session_times = _truncate_window(
                session_features, session_times, max_seq_len
            )
            n = len(session_times)
            if n < 2:
                return 30.0
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n)
            dt[1:] = np.diff(session_times)
            delta_times = (
                torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            )
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            _, mu, _, _ = self.model(features, times_t, delta_times, padding_mask)
            # Retour en minutes : exp(mu) - 1
            return max(0.0, np.expm1(mu[0, -1].item()))

    def predict_with_uncertainty(
        self,
        session_features,
        session_times,
        n_mc=50,
        mc_batch_size=10,
        max_seq_len=200,
    ):
        """MC Dropout chunké : n_mc tirages répartis en forwards de mc_batch_size.

        Compromis entre vectorisation totale (1 forward batch=n_mc, pic mémoire
        O(n_mc × S² × d) qui OOM sur grands Transformers + longues séquences) et
        purement séquentiel (n_mc forwards batch=1, lent à cause des syncs).
        Par défaut : 5 forwards de 10 → ~5× moins de mémoire pic que full-vec,
        ~10× plus rapide que séquentiel.

        max_seq_len : cap la longueur de séquence aux derniers événements pour
        rester dans la zone vue à l'entraînement (max_len=200) et borner la
        mémoire d'attention en O(S²).
        """
        self.model.train()  # dropout actif
        with torch.no_grad():
            session_features, session_times = _truncate_window(
                session_features, session_times, max_seq_len
            )
            n = len(session_times)
            if n < 2:
                return 30.0, 15.0, [30.0] * n_mc

            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n)
            dt[1:] = np.diff(session_times)
            delta_times = (
                torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            )
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)

            mu_chunks, sigma_chunks = [], []
            for start in range(0, n_mc, mc_batch_size):
                bs = min(mc_batch_size, n_mc - start)
                f_b = features.repeat(bs, 1, 1)
                t_b = times_t.repeat(bs, 1)
                dt_b = delta_times.repeat(bs, 1, 1)
                pm_b = padding_mask.repeat(bs, 1)

                _, mu, log_sigma, _ = self.model(f_b, t_b, dt_b, pm_b)
                mu_chunks.append(mu[:, -1].cpu().numpy())
                sigma_chunks.append(np.exp(log_sigma[:, -1].cpu().numpy()))

            mu_arr = np.concatenate(mu_chunks)
            sigma_arr = np.concatenate(sigma_chunks)

        preds_minutes = np.maximum(np.expm1(mu_arr), 0)
        mean_pred = preds_minutes.mean()
        epistemic = preds_minutes.std()
        aleatoric = np.mean(sigma_arr) * mean_pred  # delta method
        total_std = np.sqrt(epistemic**2 + aleatoric**2)

        return mean_pred, total_std, preds_minutes.tolist()


# ----------------------------------------------------═══════════════════════════
# 4. ÉVALUATION
# ----------------------------------------------------═══════════════════════════


def evaluate_gaussian_model(
    trainer,
    test_sessions,
    label="Model",
    with_uncertainty=False,
    n_mc=50,
    mc_batch_size=10,
    max_seq_len=200,
):
    """Évalue modèle gaussien avec/sans incertitude."""
    results = []
    desc = f"Eval {label}" + (" (MC)" if with_uncertainty else "")
    for session in tqdm(test_sessions, desc=desc, leave=False):
        times = session["times"]
        features = session["features"]
        T = session["T"]
        airport = session["airport"]

        for frac in [0.3, 0.5, 0.7, 0.9]:
            idx = int(len(times) * frac)
            if idx < 2:
                continue
            true_remaining = T - times[idx]
            if true_remaining < 0:
                continue

            if with_uncertainty:
                mean_pred, std_pred, _ = trainer.predict_with_uncertainty(
                    features[: idx + 1],
                    times[: idx + 1],
                    n_mc=n_mc,
                    mc_batch_size=mc_batch_size,
                    max_seq_len=max_seq_len,
                )
                results.append(
                    {
                        "true": true_remaining,
                        "pred": mean_pred,
                        "uncertainty": std_pred,
                        "airport": airport,
                        "airport_alert_id": session.get("alert_id"),
                        "frac": frac,
                    }
                )
            else:
                pred = trainer.predict_time_remaining(
                    features[: idx + 1],
                    times[: idx + 1],
                    max_seq_len=max_seq_len,
                )
                results.append(
                    {
                        "true": true_remaining,
                        "pred": pred,
                        "airport": airport,
                        "airport_alert_id": session.get("alert_id"),
                        "frac": frac,
                    }
                )

    df = pd.DataFrame(results)
    if len(df) == 0:
        print(f"  {label}: Pas de prédictions")
        return None

    errors = np.abs(df["true"] - df["pred"])
    mae = errors.mean()
    rmse = np.sqrt(np.mean((df["true"] - df["pred"]) ** 2))
    median = np.median(errors)
    bias = np.mean(df["pred"] - df["true"])
    p90 = np.percentile(errors, 90)

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  MAE :     {mae:.2f} min")
    print(f"  RMSE :    {rmse:.2f} min")
    print(f"  Médiane : {median:.2f} min")
    print(f"  P90 :     {p90:.2f} min")
    print(f"  Biais :   {bias:+.2f} min")

    if with_uncertainty and "uncertainty" in df.columns:
        mean_unc = df["uncertainty"].mean()
        in_1std = np.mean(errors <= df["uncertainty"])
        in_2std = np.mean(errors <= 2 * df["uncertainty"])
        print(f"  σ moyen : {mean_unc:.2f} min")
        print(f"  Cal. 1σ : {in_1std * 100:.1f}% (idéal 68.3%)")
        print(f"  Cal. 2σ : {in_2std * 100:.1f}% (idéal 95.4%)")

    print(f"{'=' * 55}")

    print("\n  Par aéroport :")
    airport_metrics = {}
    for ap, g in df.groupby("airport"):
        ap_mae = np.mean(np.abs(g["true"] - g["pred"]))
        ap_bias = np.mean(g["pred"] - g["true"])
        extra = ""
        if with_uncertainty and "uncertainty" in g.columns:
            extra = f" | σ={g['uncertainty'].mean():.2f}"
        print(
            f"    {ap:12s} | MAE={ap_mae:.2f} | Biais={ap_bias:+.2f}{extra} | n={len(g)}"
        )
        airport_metrics[ap] = {"mae": ap_mae, "bias": ap_bias, "n": len(g)}

    result = {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "median": median,
        "bias": bias,
        "p90": p90,
        "errors_df": df,
        "airport_metrics": airport_metrics,
    }
    if with_uncertainty and "uncertainty" in df.columns:
        result["mean_uncertainty"] = mean_unc
        result["calibration_1std"] = in_1std
        result["calibration_2std"] = in_2std

    return result


def run_v3_experiments(sessions, test_ratio=0.2):
    """Expériences v3 : Gaussian NLL + log-target."""

    np.random.seed(42)
    n = len(sessions)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]

    print(f"Train: {len(train_sessions)} | Test: {len(test_sessions)}\n")
    results = {}

    # ─── V3a : GRU + Gaussian NLL ─────────────────────────────────────
    print("=" * 60)
    print("  V3a : GRU + Gaussian NLL (log-space)")
    print("=" * 60)
    model_gru = GaussianHawkesGRU(input_dim=12, hidden_dim=64, dropout=0.15)
    trainer_gru = GaussianHawkesTrainer(model_gru, lr=5e-4, reg_weight=0.5)
    trainer_gru.fit(train_sessions, n_epochs=80, batch_size=32, desc="V3a GRU")
    r = evaluate_gaussian_model(
        trainer_gru, test_sessions, "GRU Gaussian (log)", with_uncertainty=True, n_mc=50
    )
    if r:
        results["gru_gaussian"] = r
    torch.save(model_gru.state_dict(), str(neural_hawkes_v2.MODEL_DIR / "hawkes_gru_gaussian.pt"))

    # ─── V3b : Transformer + Gaussian NLL ─────────────────────────────
    print("\n" + "=" * 60)
    print("  V3b : Transformer + Gaussian NLL (log-space)")
    print("=" * 60)
    model_tf = GaussianHawkesTransformer(
        input_dim=12,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.15,
    )
    trainer_tf = GaussianHawkesTrainer(model_tf, lr=5e-4, reg_weight=0.5)
    trainer_tf.fit(train_sessions, n_epochs=80, batch_size=32, desc="V3b TF")
    r = evaluate_gaussian_model(
        trainer_tf,
        test_sessions,
        "Transformer Gaussian (log)",
        with_uncertainty=True,
        n_mc=50,
    )
    if r:
        results["transformer_gaussian"] = r
    torch.save(model_tf.state_dict(), str(neural_hawkes_v2.MODEL_DIR / "hawkes_tf_gaussian.pt"))

    # ─── V3c : Transformer Large + Gaussian NLL ──────────────────────
    print("\n" + "=" * 60)
    print("  V3c : Transformer Large + Gaussian NLL (log-space)")
    print("=" * 60)
    model_tf_lg = GaussianHawkesTransformer(
        input_dim=12,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.15,
    )
    trainer_tf_lg = GaussianHawkesTrainer(model_tf_lg, lr=3e-4, reg_weight=0.5)
    trainer_tf_lg.fit(train_sessions, n_epochs=80, batch_size=32, desc="V3c TF-Large")
    r = evaluate_gaussian_model(
        trainer_tf_lg,
        test_sessions,
        "TF Large Gaussian (log)",
        with_uncertainty=True,
        n_mc=50,
    )
    if r:
        results["transformer_large_gaussian"] = r
    torch.save(model_tf_lg.state_dict(), str(neural_hawkes_v2.MODEL_DIR / "hawkes_tf_large_gaussian.pt"))

    # ─── V3d : Modèles par aéroport ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  V3d : Modèles par aéroport (Transformer Gaussian)")
    print("=" * 60)
    airports = set(s["airport"] for s in sessions)
    per_airport_errors = []

    for airport in sorted(airports):
        ap_train = [s for s in train_sessions if s["airport"] == airport]
        ap_test = [s for s in test_sessions if s["airport"] == airport]
        print(f"\n  --- {airport} : {len(ap_train)} train / {len(ap_test)} test ---")

        if len(ap_train) < 10 or len(ap_test) < 3:
            print("  ⚠ Fallback au modèle global")
            r = evaluate_gaussian_model(trainer_tf, ap_test, f"Global → {airport}")
            if r:
                per_airport_errors.append(r["errors_df"])
            continue

        model_ap = GaussianHawkesTransformer(
            input_dim=12,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.15,
        )
        trainer_ap = GaussianHawkesTrainer(model_ap, lr=5e-4, reg_weight=0.5)
        trainer_ap.fit(
            ap_train,
            n_epochs=80,
            batch_size=min(16, len(ap_train)),
            desc=f"V3d {airport}",
        )
        r = evaluate_gaussian_model(trainer_ap, ap_test, f"Local → {airport}")
        if r:
            per_airport_errors.append(r["errors_df"])
        torch.save(
            model_ap.state_dict(),
            str(neural_hawkes_v2.MODEL_DIR / f"hawkes_gaussian_{airport.lower()}.pt"),
        )

    if per_airport_errors:
        combined = pd.concat(per_airport_errors, ignore_index=True)
        mae_c = np.mean(np.abs(combined["true"] - combined["pred"]))
        rmse_c = np.sqrt(np.mean((combined["true"] - combined["pred"]) ** 2))
        bias_c = np.mean(combined["pred"] - combined["true"])
        med_c = np.median(np.abs(combined["true"] - combined["pred"]))
        p90_c = np.percentile(np.abs(combined["true"] - combined["pred"]), 90)
        print(f"\n  Ensemble par aéroport → MAE: {mae_c:.2f} | Biais: {bias_c:+.2f}")
        results["per_airport_gaussian"] = {
            "label": "Ensemble par aéroport (Gaussian)",
            "mae": mae_c,
            "rmse": rmse_c,
            "median": med_c,
            "bias": bias_c,
            "p90": p90_c,
            "errors_df": combined,
        }

    # ─── Résumé ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("  RÉSUMÉ FINAL PHASE 2bis (toutes itérations)")
    print("=" * 75)
    print(
        f"  {'Variante':40s} | {'MAE':>6s} | {'RMSE':>6s} | {'Méd.':>6s} | {'Biais':>7s} | {'P90':>6s} | {'σ':>5s} | {'1σ':>5s}"
    )
    print("-" * 75)

    # Rappel Phase 2
    print(
        f"  {'[Réf] GRU v1 (Phase 2)':40s} | {'22.11':>6s} | {'32.50':>6s} | {'18.00':>6s} | {'-1.61':>7s} | {'40.12':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print(
        f"  {'[Réf] XGBoost AFT (Phase 2)':40s} | {'57.28':>6s} | {'90.69':>6s} | {'33.18':>6s} | {'-26.74':>7s} | {'149.8':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print(
        f"  {'[Réf] Baseline 30 min':40s} | {'60.26':>6s} | {'---':>6s} | {'---':>6s} | {'-52.32':>7s} | {'---':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print("-" * 75)
    # v2 rappel
    print(
        f"  {'[v2] GRU multi-tâche':40s} | {'26.59':>6s} | {'33.67':>6s} | {'24.18':>6s} | {'+11.04':>7s} | {'45.05':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print(
        f"  {'[v2] Transformer multi-tâche':40s} | {'29.17':>6s} | {'35.55':>6s} | {'28.64':>6s} | {'+14.10':>7s} | {'44.86':>6s} | {'---':>5s} | {'---':>5s}"
    )
    print("-" * 75)

    # v3
    for key in [
        "gru_gaussian",
        "transformer_gaussian",
        "transformer_large_gaussian",
        "per_airport_gaussian",
    ]:
        if key in results:
            r = results[key]
            unc = (
                f"{r.get('mean_uncertainty', 0):.1f}"
                if "mean_uncertainty" in r
                else "---"
            )
            cal1 = (
                f"{r.get('calibration_1std', 0) * 100:.0f}%"
                if "calibration_1std" in r
                else "---"
            )
            print(
                f"  {r['label']:40s} | {r['mae']:6.2f} | {r.get('rmse', 0):6.2f} | {r.get('median', 0):6.2f} | {r['bias']:+7.2f} | {r.get('p90', 0):6.2f} | {unc:>5s} | {cal1:>5s}"
            )

    print("=" * 75)

    # Sauvegarde
    for key, r in results.items():
        if "errors_df" in r and r["errors_df"] is not None:
            r["errors_df"].to_parquet(f"data/errors_{key}.parquet", index=False)
    summary = {}
    for key, r in results.items():
        summary[key] = {
            k: v for k, v in r.items() if k not in ("errors_df", "airport_metrics")
        }
    with open(str(neural_hawkes_v2.MODEL_DIR / "phase2bis_v3_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    start_exec = time.perf_counter()

    print("=" * 60)
    print("  PHASE 2bis - Itération 2 : Gaussian NLL + log-target")
    print("=" * 60)

    print("\nLoading data...")
    df = data_loader.load_raw()
    alerts = data_loader.load_alerts(df)

    print("Preparing sessions...")
    sessions = neural_hawkes_v2.prepare_sessions_v2(alerts)
    print(f"  {len(sessions)} sessions\n")

    results = run_v3_experiments(sessions)

    print(f"\nResultats sauvegardes dans {neural_hawkes_v2.MODEL_DIR}/")
    print("Termine !")

    elapsed = time.perf_counter() - start_exec
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\nTemps total d'exécution : {hours:02d}h {minutes:02d}m {seconds:05.2f}s")
