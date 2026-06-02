"""
neural_hawkes_v2.py - Phase 2bis : Neural Hawkes Process amélioré.

Architecture Multi-Tâche :
  - Tâche 1 (NLL) : apprendre la dynamique temporelle du processus ponctuel
  - Tâche 2 (Régression) : prédire directement le temps restant depuis le hidden state

Améliorations par rapport à la v1 (GRU, 4 features) :
  1. Features enrichies (12 dim) : azimuth, centroïde mobile, densité, inter-arrival CG
  2. Architecture Transformer avec encodage positionnel continu
  3. Tête de régression directe (au lieu d'extrapolation exponentielle)
  4. Modèles spécialisés par aéroport
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import math
import json
from pathlib import Path
import data_loader
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[neural_hawkes_v2] Device: {DEVICE}")

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------═══════════════════════════
# 1. PREPARATION DES DONNÉES (features enrichies)
# ----------------------------------------------------═══════════════════════════

def prepare_sessions_v2(df: pd.DataFrame):
    """
    Prépare les sessions avec features enrichies (12 dim) + cible time_to_end.
    
    Features par événement :
      0  icloud, 1  amplitude_norm, 2  dist_norm, 3  maxis_norm,
      4  azimuth_sin, 5  azimuth_cos,
      6  dt_since_last, 7  dt_since_last_cg, 8  density_5min,
      9  ic_ratio_recent, 10  dist_to_centroid, 11  dist_trend
    """
    sessions = []
    
    for (airport, alert_id), group in df.groupby(["airport", "airport_alert_id"]):
        group = group.sort_values("date").copy()
        
        cg = group[~group["icloud"]].copy()
        if len(cg) < 2:
            continue
        
        start = group["date"].iloc[0]
        last_cg = cg["date"].iloc[-1]
        
        times = (group["date"] - start).dt.total_seconds().values / 60.0
        cg_times = (cg["date"] - start).dt.total_seconds().values / 60.0
        
        n = len(group)
        T = (last_cg - start).total_seconds() / 60.0
        
        # --- Time to end pour chaque event (cible de régression) ---
        time_to_end = np.maximum(T - times, 0.0)
        
        # --- Features ---
        f_icloud = group["icloud"].astype(float).values
        f_amp = group["amplitude"].values / 100.0
        f_dist = group["dist"].values / 50.0
        f_maxis = group["maxis"].values / 10.0
        
        az_rad = np.deg2rad(group["azimuth"].values)
        f_az_sin = np.sin(az_rad)
        f_az_cos = np.cos(az_rad)
        
        f_dt = np.zeros(n)
        if n > 1:
            f_dt[1:] = np.diff(times)
        f_dt = np.clip(f_dt / 30.0, 0, 1)
        
        all_dates = group["date"].values
        f_dt_cg = np.zeros(n)
        last_cg_seen = None
        for i in range(n):
            if not group["icloud"].iloc[i]:
                last_cg_seen = all_dates[i]
                f_dt_cg[i] = 0.0
            elif last_cg_seen is not None:
                f_dt_cg[i] = (all_dates[i] - last_cg_seen) / np.timedelta64(1, 'm')
        f_dt_cg = np.clip(f_dt_cg / 30.0, 0, 1)
        
        f_density = np.zeros(n)
        for i in range(n):
            window_start = times[i] - 5.0
            f_density[i] = np.sum((times >= window_start) & (times <= times[i]))
        f_density = f_density / 50.0
        
        f_ic_ratio = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - 9)
            window = f_icloud[start_idx:i+1]
            f_ic_ratio[i] = np.mean(window)
        
        lats = group["lat"].values
        lons = group["lon"].values
        f_dist_centroid = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - 9)
            clat = np.mean(lats[start_idx:i+1])
            clon = np.mean(lons[start_idx:i+1])
            f_dist_centroid[i] = np.sqrt(((lats[i] - clat) * 111)**2 + 
                                          ((lons[i] - clon) * 111 * np.cos(np.radians(clat)))**2)
        f_dist_centroid = np.clip(f_dist_centroid / 20.0, 0, 1)
        
        f_dist_trend = np.zeros(n)
        for i in range(1, n):
            start_idx = max(0, i - 4)
            recent_dist = f_dist[start_idx:i+1]
            if len(recent_dist) >= 2:
                f_dist_trend[i] = recent_dist[-1] - recent_dist[0]
        
        features = np.column_stack([
            f_icloud, f_amp, f_dist, f_maxis,
            f_az_sin, f_az_cos,
            f_dt, f_dt_cg, f_density, f_ic_ratio,
            f_dist_centroid, f_dist_trend
        ])
        
        sessions.append({
            "airport": airport,
            "alert_id": alert_id,
            "times": times,
            "cg_times": cg_times,
            "features": features,
            "time_to_end": time_to_end,
            "T": T,
            "lons": lons,
            "lats": lats,
            "start_time": start,
            "session_key": f"{airport}_{alert_id}",
        })
    
    return sessions


# ----------------------------------------------------═══════════════════════════
# 2. CONTINUOUS-TIME POSITIONAL ENCODING
# ----------------------------------------------------═══════════════════════════

class ContinuousPositionalEncoding(nn.Module):
    """
    Encodage positionnel continu basé sur les timestamps réels (en minutes).
    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, timestamps):
        t = timestamps.unsqueeze(-1)
        pe = torch.zeros(*timestamps.shape, self.d_model, device=timestamps.device)
        pe[..., 0::2] = torch.sin(t * self.div_term)
        pe[..., 1::2] = torch.cos(t * self.div_term)
        return pe


# ----------------------------------------------------═══════════════════════════
# 3. NEURAL HAWKES TRANSFORMER (Multi-Tâche)
# ----------------------------------------------------═══════════════════════════

class NeuralHawkesTransformer(nn.Module):
    """
    Neural Hawkes Process Transformer — architecture multi-tâche.
    
    Deux têtes de sortie :
      - intensity_head : prédit λ(t) pour la NLL du processus ponctuel
      - regression_head : prédit directement le temps restant (minutes)
    
    Le backbone Transformer apprend la dynamique temporelle via les deux objectifs.
    """
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = ContinuousPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Tête 1 : intensité du processus ponctuel
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.softplus = nn.Softplus()
        
        # Tête 2 : régression directe du temps restant
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # temps restant >= 0
        )
        
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
        
        # Régression temps restant
        time_pred = self.regression_head(h).squeeze(-1)
        
        return intensity, time_pred, h


class NeuralHawkesGRUv2(nn.Module):
    """GRU v2 multi-tâche avec features enrichies."""
    def __init__(self, input_dim=12, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.softplus = nn.Softplus()
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        gru_out, _ = self.gru(event_features)
        combined = torch.cat([gru_out, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_layer(combined)).squeeze(-1)
        time_pred = self.regression_head(gru_out).squeeze(-1)
        return intensity, time_pred, gru_out


# ----------------------------------------------------═══════════════════════════
# 4. DATASET & TRAINING (Multi-Tâche)
# ----------------------------------------------------═══════════════════════════

class HawkesDatasetV2(Dataset):
    """Dataset avec time_to_end pour la régression."""
    def __init__(self, sessions, max_len=200):
        self.sessions = []
        self.max_len = max_len
        self.feat_dim = None
        
        for s in sessions:
            if len(s["times"]) < 3:
                continue
            n = min(len(s["times"]), max_len)
            self.sessions.append({
                "times": s["times"][:n],
                "features": s["features"][:n],
                "time_to_end": s["time_to_end"][:n],
                "n": n,
                "airport": s["airport"]
            })
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
        mask = np.zeros(self.max_len)
        
        times[:n] = s["times"]
        features[:n] = s["features"]
        time_to_end[:n] = s["time_to_end"]
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
            "mask": torch.FloatTensor(mask),
            "padding_mask": torch.BoolTensor(padding_mask),
            "n": n
        }


class NeuralHawkesTrainer:
    """Trainer multi-tâche : NLL + régression."""
    
    def __init__(self, model, lr=5e-4, reg_weight=1.0, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.device = device
        self.reg_weight = reg_weight
        self.huber = nn.HuberLoss(delta=20.0, reduction='none')
    
    def combined_loss(self, intensity, time_pred, delta_times, time_to_end, mask):
        """
        Loss combinée :
          L = NLL (processus ponctuel) + λ * Huber(time_pred, time_to_end)
        """
        # NLL
        log_lambda = torch.log(intensity + 1e-8) * mask
        compensator = intensity * delta_times.squeeze(-1) * mask
        nll = (-log_lambda.sum() + compensator.sum()) / (mask.sum() + 1e-8)
        
        # Régression Huber (robuste aux outliers)
        reg_loss = (self.huber(time_pred, time_to_end) * mask).sum() / (mask.sum() + 1e-8)
        
        return nll + self.reg_weight * reg_loss, nll.item(), reg_loss.item()
    
    def fit(self, train_sessions, n_epochs=50, batch_size=32, verbose=True, desc="Train"):
        dataset = HawkesDatasetV2(train_sessions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        epoch_bar = tqdm(range(n_epochs), desc=desc, disable=not verbose, leave=True)
        for epoch in epoch_bar:
            total_nll = 0
            total_reg = 0
            n_batches = 0

            for batch in loader:
                features = batch["features"].to(self.device)
                times = batch["times"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                time_to_end = batch["time_to_end"].to(self.device)
                mask = batch["mask"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                intensity, time_pred, _ = self.model(features, times, delta_times, padding_mask)
                loss, nll_val, reg_val = self.combined_loss(
                    intensity, time_pred, delta_times, time_to_end, mask
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_nll += nll_val
                total_reg += reg_val
                n_batches += 1

            avg_nll = total_nll / n_batches
            avg_reg = total_reg / n_batches
            self.scheduler.step()

            if avg_reg < best_loss - 1e-4:
                best_loss = avg_reg
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            epoch_bar.set_postfix(
                nll=f"{avg_nll:.4f}",
                reg=f"{avg_reg:.4f}",
                best=f"{best_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.1e}",
                pat=patience_counter,
            )

            if patience_counter >= 15:
                epoch_bar.write(f"  Early stopping at epoch {epoch+1}")
                epoch_bar.close()
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self
    
    def predict_time_remaining(self, session_features, session_times):
        """Prédit le temps restant via la tête de régression directe."""
        self.model.eval()
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0
            
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n)
            dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            
            _, time_pred, _ = self.model(features, times_t, delta_times, padding_mask)
            return time_pred[0, -1].item()


# ----------------------------------------------------═══════════════════════════
# 5. ÉVALUATION COMPARATIVE
# ----------------------------------------------------═══════════════════════════

def evaluate_model(trainer, test_sessions, label="Model"):
    """Évalue un modèle sur les sessions de test."""
    errors = []
    for session in test_sessions:
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
            pred_remaining = trainer.predict_time_remaining(
                features[:idx+1], times[:idx+1]
            )
            errors.append({
                "true": true_remaining, "pred": pred_remaining,
                "airport": airport, "airport_alert_id": session.get("alert_id"), "frac": frac
            })
    
    df = pd.DataFrame(errors)
    if len(df) == 0:
        print(f"  {label}: Pas de prédictions")
        return None
    
    mae = np.mean(np.abs(df["true"] - df["pred"]))
    rmse = np.sqrt(np.mean((df["true"] - df["pred"])**2))
    median = np.median(np.abs(df["true"] - df["pred"]))
    bias = np.mean(df["pred"] - df["true"])
    p90 = np.percentile(np.abs(df["true"] - df["pred"]), 90)
    
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  MAE :     {mae:.2f} min")
    print(f"  RMSE :    {rmse:.2f} min")
    print(f"  Médiane : {median:.2f} min")
    print(f"  P90 :     {p90:.2f} min")
    print(f"  Biais :   {bias:+.2f} min")
    print(f"{'='*55}")
    
    print(f"\n  Par aéroport :")
    airport_metrics = {}
    for ap, g in df.groupby("airport"):
        ap_mae = np.mean(np.abs(g["true"] - g["pred"]))
        ap_bias = np.mean(g["pred"] - g["true"])
        print(f"    {ap:12s} | MAE={ap_mae:.2f} | Biais={ap_bias:+.2f} | n={len(g)}")
        airport_metrics[ap] = {"mae": ap_mae, "bias": ap_bias, "n": len(g)}
    
    return {
        "label": label, "mae": mae, "rmse": rmse, "median": median,
        "bias": bias, "p90": p90, "errors_df": df,
        "airport_metrics": airport_metrics
    }


def evaluate_all_variants(sessions, test_ratio=0.2):
    """Entraîne et évalue toutes les variantes."""
    
    np.random.seed(42)
    n = len(sessions)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]
    
    print(f"Train: {len(train_sessions)} sessions | Test: {len(test_sessions)} sessions\n")
    
    all_results = {}
    
    # ─── Variante 1 : GRU v2 (features enrichies) ─────────────────────
    print("=" * 60)
    print("  VARIANTE 1 : GRU v2 (12 features, multi-tâche)")
    print("=" * 60)
    model_gru = NeuralHawkesGRUv2(input_dim=12, hidden_dim=64, dropout=0.1)
    trainer_gru = NeuralHawkesTrainer(model_gru, lr=5e-4, reg_weight=1.0)
    trainer_gru.fit(train_sessions, n_epochs=80, batch_size=32, desc="V1 GRU")
    r = evaluate_model(trainer_gru, test_sessions, "GRU v2 (12 feat., multi-tâche)")
    if r: all_results["gru_v2"] = r
    torch.save(model_gru.state_dict(), str(MODEL_DIR / "neural_hawkes_gru_v2.pt"))
    
    # ─── Variante 2 : Transformer ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VARIANTE 2 : Transformer (d=64, 4h, 3L, multi-tâche)")
    print("=" * 60)
    model_tf = NeuralHawkesTransformer(
        input_dim=12, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.1
    )
    trainer_tf = NeuralHawkesTrainer(model_tf, lr=5e-4, reg_weight=1.0)
    trainer_tf.fit(train_sessions, n_epochs=80, batch_size=32, desc="V2 TF")
    r = evaluate_model(trainer_tf, test_sessions, "Transformer (d=64, 3L)")
    if r: all_results["transformer"] = r
    torch.save(model_tf.state_dict(), str(MODEL_DIR / "neural_hawkes_transformer.pt"))
    
    # ─── Variante 3 : Transformer large ───────────────────────────────
    print("\n" + "=" * 60)
    print("  VARIANTE 3 : Transformer Large (d=128, 8h, 4L)")
    print("=" * 60)
    model_tf_lg = NeuralHawkesTransformer(
        input_dim=12, d_model=128, nhead=8, num_layers=4,
        dim_feedforward=256, dropout=0.15
    )
    trainer_tf_lg = NeuralHawkesTrainer(model_tf_lg, lr=3e-4, reg_weight=1.0)
    trainer_tf_lg.fit(train_sessions, n_epochs=80, batch_size=32, desc="V3 TF-Large")
    r = evaluate_model(trainer_tf_lg, test_sessions, "Transformer Large (d=128, 4L)")
    if r: all_results["transformer_large"] = r
    torch.save(model_tf_lg.state_dict(), str(MODEL_DIR / "neural_hawkes_transformer_large.pt"))
    
    # ─── Variante 4 : Modèles par aéroport ────────────────────────────
    print("\n" + "=" * 60)
    print("  VARIANTE 4 : Modèles par aéroport (Transformer)")
    print("=" * 60)
    
    airports = set(s["airport"] for s in sessions)
    per_airport_errors = []
    
    for airport in sorted(airports):
        ap_train = [s for s in train_sessions if s["airport"] == airport]
        ap_test = [s for s in test_sessions if s["airport"] == airport]
        
        print(f"\n  --- {airport} : {len(ap_train)} train / {len(ap_test)} test ---")
        
        if len(ap_train) < 10 or len(ap_test) < 3:
            print(f"  ⚠ Trop peu de sessions, on utilise le modèle global")
            r = evaluate_model(trainer_tf, ap_test, f"Global → {airport}")
            if r and r["errors_df"] is not None:
                per_airport_errors.append(r["errors_df"])
            continue
        
        model_ap = NeuralHawkesTransformer(
            input_dim=12, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=128, dropout=0.1
        )
        trainer_ap = NeuralHawkesTrainer(model_ap, lr=5e-4, reg_weight=1.0)
        trainer_ap.fit(ap_train, n_epochs=80, batch_size=min(16, len(ap_train)), desc=f"V4 {airport}")
        r = evaluate_model(trainer_ap, ap_test, f"Local → {airport}")
        
        if r and r["errors_df"] is not None:
            per_airport_errors.append(r["errors_df"])
        torch.save(model_ap.state_dict(), 
                   str(MODEL_DIR / f"neural_hawkes_tf_{airport.lower()}.pt"))
    
    if per_airport_errors:
        combined = pd.concat(per_airport_errors, ignore_index=True)
        mae_combined = np.mean(np.abs(combined["true"] - combined["pred"]))
        rmse_combined = np.sqrt(np.mean((combined["true"] - combined["pred"])**2))
        bias_combined = np.mean(combined["pred"] - combined["true"])
        med_combined = np.median(np.abs(combined["true"] - combined["pred"]))
        p90_combined = np.percentile(np.abs(combined["true"] - combined["pred"]), 90)
        print(f"\n  Ensemble par aéroport → MAE: {mae_combined:.2f} | Biais: {bias_combined:+.2f}")
        all_results["per_airport"] = {
            "label": "Ensemble par aéroport",
            "mae": mae_combined, "rmse": rmse_combined,
            "median": med_combined, "bias": bias_combined, "p90": p90_combined,
            "errors_df": combined
        }
    
    # ─── Résumé comparatif ─────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  RÉSUMÉ COMPARATIF PHASE 2bis")
    print("=" * 70)
    print(f"  {'Variante':35s} | {'MAE':>7s} | {'RMSE':>7s} | {'Méd.':>7s} | {'Biais':>8s} | {'P90':>7s}")
    print("-" * 70)
    
    print(f"  {'[Phase 2] GRU v1 (4 feat.)':35s} | {'22.11':>7s} | {'32.50':>7s} | {'18.00':>7s} | {'-1.61':>8s} | {'40.12':>7s}")
    print(f"  {'[Phase 2] Baseline 30 min':35s} | {'60.26':>7s} | {'---':>7s} | {'---':>7s} | {'-52.32':>8s} | {'---':>7s}")
    print("-" * 70)
    
    for key in ["gru_v2", "transformer", "transformer_large", "per_airport"]:
        if key in all_results:
            r = all_results[key]
            print(f"  {r['label']:35s} | {r['mae']:7.2f} | {r.get('rmse',0):7.2f} | {r.get('median',0):7.2f} | {r['bias']:+8.2f} | {r.get('p90',0):7.2f}")
    
    print("=" * 70)
    
    # Sauvegarder
    results_summary = {}
    for key, r in all_results.items():
        results_summary[key] = {k: v for k, v in r.items() if k not in ("errors_df", "airport_metrics")}
    with open(str(MODEL_DIR / "phase2bis_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    for key, r in all_results.items():
        if "errors_df" in r and r["errors_df"] is not None:
            r["errors_df"].to_parquet(
                str(Path(__file__).parent.parent / "data" / f"errors_{key}.parquet"), index=False
            )
    
    return all_results, trainer_tf, trainer_tf_lg


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 2bis - Neural Hawkes Process : Améliorations")
    print("=" * 60)
    
    print("\nLoading data...")
    df = data_loader.load_raw()
    alerts = data_loader.load_alerts(df)
    
    print("Preparing enriched sessions (12 features)...")
    sessions = prepare_sessions_v2(alerts)
    print(f"  {len(sessions)} sessions préparées")
    print(f"  Features par événement : {sessions[0]['features'].shape[1]}")
    
    all_results, best_trainer, _ = evaluate_all_variants(sessions)
    
    print(f"\nRésultats sauvegardés dans {MODEL_DIR}/")
    print("Terminé !")
