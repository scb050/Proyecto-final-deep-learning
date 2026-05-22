"""Temporal Fusion Transformer — implementación completa.

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
Transformers for interpretable multi-horizon time series forecasting.
International Journal of Forecasting, 37(4), 1748–1764.
https://doi.org/10.1016/j.ijforecast.2021.03.012

Adaptaciones respecto al paper original:
- Input único: x (B, lookback, n_features) — sin covariables estáticas/futuras.
- Pérdida MSE punto (no quantile loss); parámetro `quantiles` se acepta para
  compatibilidad con tft.yaml pero no afecta la salida.
- Decoder basado en cross-attention con queries aprendibles por paso de horizonte,
  lo que permite horizon ≠ lookback de forma general.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseForecaster


class _GRN(nn.Module):
    """Gated Residual Network: bloque fundamental del TFT (sec. 3.3 del paper).

    out = LayerNorm(x + Dropout(ELU(fc1(x)) * sigmoid(fc2_gate(x))))
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size * 2)   # produce valor + gate en un solo paso
        self.norm = nn.LayerNorm(size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h1, h2 = self.fc2(h).chunk(2, dim=-1)  # valor, gate
        return self.norm(x + self.drop(h1 * h2.sigmoid()))


class _VSN(nn.Module):
    """Variable Selection Network: gating suave sobre los n_features de entrada.

    Cada feature se proyecta individualmente a hidden_size; un contexto global
    (combinación lineal de todos los features) produce pesos softmax que
    ponderan esas proyecciones individuales.
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.n_features = n_features
        # Proyección individual: una Linear(1 → H) por feature
        self.feat_proj = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(n_features)]
        )
        # Red de contexto global → pesos de selección
        self.ctx_proj = nn.Linear(n_features, hidden_size)
        self.weight_net = nn.Linear(hidden_size, n_features)
        self.grn = _GRN(hidden_size, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        ctx = self.ctx_proj(x)                           # (B, L, H)
        weights = self.weight_net(ctx).softmax(dim=-1)   # (B, L, F)

        # Proyecciones individuales: (B, L, F, H)
        feats = torch.stack(
            [self.feat_proj[i](x[..., i : i + 1]) for i in range(self.n_features)],
            dim=2,
        )
        # Suma ponderada: (B, L, H)
        out = (weights.unsqueeze(-1) * feats).sum(dim=2)
        return self.grn(self.drop(out))


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer (Lim et al., 2021).

    Flujo:
        x (B, L, F)
        → VSN                        — selección soft de variables por paso
        → LSTM encoder (2 capas)     — dependencias locales + estado oculto
        → [residual + LayerNorm + GRN]
        → Multi-head Self-Attention  — dependencias largas en lookback
        → [residual + LayerNorm + GRN]
        → Cross-Attention (queries aprendibles por horizonte)
        → [residual + LayerNorm + GRN]
        → Linear(H → n_targets)
        → ŷ (B, horizon, n_targets)
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        hidden_size: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.1,
        n_static_categorical: int = 4,
        n_static_real: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            dropout=dropout,
            n_static_categorical=n_static_categorical,
            n_static_real=n_static_real,
            quantiles=tuple(quantiles),
        )

        H = hidden_size

        # 1. Variable Selection Network
        self.vsn = _VSN(n_features, H, dropout)

        # 2. LSTM encoder local (2 capas)
        self.encoder_lstm = nn.LSTM(
            input_size=H,
            hidden_size=H,
            num_layers=2,
            dropout=dropout,   # entre capas 1→2; ignorado si num_layers=1
            batch_first=True,
        )
        self.encoder_norm = nn.LayerNorm(H)
        self.encoder_grn = _GRN(H, dropout)

        # 3. Multi-head Self-Attention — dependencias largas en el lookback
        self.self_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(H)
        self.attn_grn = _GRN(H, dropout)

        # 4. Decoder: un query aprendible por paso de horizonte
        #    Cada query atiende a todo el contexto encoder via cross-attention.
        self.decoder_queries = nn.Parameter(torch.randn(horizon, H) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(H)
        self.decoder_grn = _GRN(H, dropout)

        # 5. Proyección final a espacio de targets
        self.output_proj = nn.Linear(H, n_targets)

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, lookback, n_features)
        Returns:
            ŷ: (B, horizon, n_targets)
        """
        B = x.size(0)

        # --- 1. Variable Selection ---
        vsn_out = self.vsn(x)                               # (B, L, H)

        # --- 2. LSTM encoder ---
        lstm_out, _ = self.encoder_lstm(vsn_out)            # (B, L, H)
        enc = self.encoder_norm(vsn_out + lstm_out)         # residual add & norm
        enc = self.encoder_grn(enc)                         # (B, L, H)

        # --- 3. Multi-head Self-Attention ---
        attn_out, _ = self.self_attn(enc, enc, enc)         # (B, L, H)
        ctx = self.attn_norm(enc + attn_out)                # residual add & norm
        ctx = self.attn_grn(ctx)                            # (B, L, H)

        # --- 4. Cross-Attention decoder ---
        # queries: un vector aprendible por cada paso del horizonte
        queries = self.decoder_queries.unsqueeze(0).expand(B, -1, -1)  # (B, horizon, H)
        dec_out, _ = self.cross_attn(queries, ctx, ctx)                # (B, horizon, H)
        dec_out = self.decoder_norm(queries + dec_out)                  # residual
        dec_out = self.decoder_grn(dec_out)                            # (B, horizon, H)

        # --- 5. Proyección ---
        return self.output_proj(dec_out)                               # (B, horizon, n_targets)
