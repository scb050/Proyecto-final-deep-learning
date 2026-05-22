"""Transformer / Informer — implementación completa.

Vaswani et al. (2017) y Zhou et al. (2021, Informer) para forecasting multistep.
La variante se selecciona mediante el parámetro `variant`:

    variant="vanilla"   → encoder-decoder estándar, self-attention O(L²).
    variant="informer"  → encoder ProbSparse O(L log L) con distilling progresivo
                          (Zhou et al., 2021, sec. 3.2–3.3); decoder vanilla con
                          cross-attention al contexto distilado.

Ambas variantes comparten embedding de entrada, positional encoding, token de
decoder aprendible y proyección de salida. La salida es punto (MSE), no
distribución cuantílica.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseForecaster


# ──────────────────────────────────────────────────── componentes compartidos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ───────────────────────────────────────────────── Informer: ProbSparse + distilling


class _ProbSparseAttention(nn.Module):
    """ProbSparse self-attention (Zhou et al., 2021, sec. 3.2).

    Para cada query se calcula una medida de dispersión
        M(q_i, K) = max_j score_ij – mean_j score_ij
    aproximada sobre u_s claves muestreadas al azar. Solo las top-u queries
    más "activas" reciben atención completa sobre K; el resto usa V.mean()
    como aproximación lazy, reduciendo la complejidad de O(L²) a O(L log L).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, factor: int) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model debe ser divisible por nhead"
        self.nhead  = nhead
        self.d_head = d_model // nhead
        self.factor = factor
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # query/key/value: (B, L, D)
        B, L_Q, _ = query.shape
        L_K = key.size(1)
        H, d = self.nhead, self.d_head

        Q = self.q_proj(query).view(B, L_Q, H, d).transpose(1, 2)    # (B,H,L_Q,d)
        K = self.k_proj(key).view(B, L_K, H, d).transpose(1, 2)      # (B,H,L_K,d)
        V = self.v_proj(value).view(B, L_K, H, d).transpose(1, 2)    # (B,H,L_K,d)

        # Cuántas queries seleccionamos y cuántas claves muestreamos
        u   = max(1, min(self.factor * math.ceil(math.log(L_K + 1)), L_Q))
        u_s = max(1, min(self.factor * math.ceil(math.log(L_K + 1)), L_K))

        # Aproximar M(q_i, K) con u_s claves aleatorias
        idx_s = torch.randint(0, L_K, (u_s,), device=query.device)
        K_s   = K[:, :, idx_s, :]                                      # (B,H,u_s,d)
        sc_s  = torch.matmul(Q, K_s.transpose(-2, -1)) / math.sqrt(d) # (B,H,L_Q,u_s)
        M     = sc_s.max(dim=-1).values - sc_s.mean(dim=-1)           # (B,H,L_Q)

        # Top-u queries por score de dispersión
        _, top_idx = M.topk(u, dim=-1, sorted=False)                  # (B,H,u)
        idx_exp    = top_idx.unsqueeze(-1).expand(-1, -1, -1, d)      # (B,H,u,d)

        # Atención completa para las queries seleccionadas
        Q_top  = Q.gather(2, idx_exp)                                  # (B,H,u,d)
        scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(d)  # (B,H,u,L_K)
        V_top  = torch.matmul(self.drop(scores.softmax(dim=-1)), V)   # (B,H,u,d)

        # Contexto: V_top donde se seleccionó, V.mean() el resto
        # scatter no-in-place para ser safe con autograd
        ctx_sel = torch.zeros(B, H, L_Q, d, device=query.device, dtype=Q.dtype)
        ctx_sel = ctx_sel.scatter(2, idx_exp, V_top)                  # (B,H,L_Q,d)
        sel_mask = torch.zeros(B, H, L_Q, dtype=torch.bool, device=query.device)
        sel_mask.scatter_(2, top_idx, True)                            # bool: sin grad
        context = torch.where(
            sel_mask.unsqueeze(-1),
            ctx_sel,
            V.mean(dim=2, keepdim=True).expand(B, H, L_Q, d),
        )                                                              # (B,H,L_Q,d)

        out = context.transpose(1, 2).reshape(B, L_Q, H * d)          # (B,L_Q,D)
        return self.out_proj(out)


class _Distilling(nn.Module):
    """Conv1d(k=3) + ELU + MaxPool(k=3, s=2) → reduce L a ⌈L/2⌉."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                             # (B,D,L)
        x = self.pool(F.elu(self.norm(self.conv(x))))     # (B,D,⌈L/2⌉)
        return x.transpose(1, 2)                          # (B,⌈L/2⌉,D)


class _InformerEncoderLayer(nn.Module):
    """ProbSparse self-attention + FFN + distilling (solo si no es la última capa)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        factor: int,
        distill: bool,
    ) -> None:
        super().__init__()
        self.self_attn = _ProbSparseAttention(d_model, nhead, dropout, factor)
        self.ff1   = nn.Linear(d_model, dim_feedforward)
        self.ff2   = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = F.gelu if activation == "gelu" else F.relu
        self.distill = _Distilling(d_model) if distill else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ff2(self.drop(self.act(self.ff1(x))))))
        if self.distill is not None:
            x = self.distill(x)
        return x


class _InformerEncoder(nn.Module):
    """Stack de InformerEncoderLayers con distilling progresivo L → L/2 → L/4.

    El distilling se aplica tras cada capa excepto la última.
    Con num_layers=3: L=168 → 84 → 42 (salida del encoder).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        factor: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _InformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, factor,
                distill=(i < num_layers - 1),
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x                                          # (B, L', D)


# ──────────────────────────────────────────────────────────── TransformerForecaster


class TransformerForecaster(BaseForecaster):
    """Transformer / Informer para forecasting multistep.

    Entrada:  x  (B, lookback, n_features)
    Salida:   ŷ  (B, horizon, n_targets)

    El decoder es siempre generativo (one-shot): un token aprendible de
    forma (1, horizon, n_targets) inicializa cada paso del horizonte en
    paralelo, sin bucle autoregresivo.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        variant: str = "vanilla",
        prob_sparse_factor: int = 5,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            lookback=lookback,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            variant=variant,
            prob_sparse_factor=prob_sparse_factor,
        )
        self.variant = variant

        # ── capas compartidas ──────────────────────────────────────────
        self.input_proj  = nn.Linear(n_features, d_model)
        self.target_proj = nn.Linear(n_targets, d_model)
        self.pos_enc     = _PositionalEncoding(d_model)
        self.head        = nn.Linear(d_model, n_targets)
        # Token aprendible: inicializa los horizon pasos del decoder
        self.tgt_token   = nn.Parameter(torch.zeros(1, horizon, n_targets))

        # ── encoder + decoder según variante ──────────────────────────
        if variant == "vanilla":
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )

        elif variant == "informer":
            # Encoder: ProbSparse self-attention + distilling progresivo
            self.informer_encoder = _InformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                factor=prob_sparse_factor,
            )
            # Decoder: cross-attention vanilla sobre el contexto distilado
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                ),
                num_layers=num_decoder_layers,
            )

        else:
            raise ValueError(
                f"variant desconocida: {variant!r}. Opciones: 'vanilla', 'informer'."
            )

    # ─────────────────────────────────────────────────────────────── forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, lookback, n_features)
        Returns:
            ŷ: (B, horizon, n_targets)
        """
        B = x.size(0)

        # ── embedding de entrada ──
        src = self.pos_enc(self.input_proj(x))              # (B, L, D)

        # ── token de decoder → proyección → pos_enc ──
        tgt = self.pos_enc(
            self.target_proj(self.tgt_token.expand(B, -1, -1))
        )                                                   # (B, horizon, D)

        # Máscara causal: cada paso del decoder solo atiende a posiciones anteriores
        causal = nn.Transformer.generate_square_subsequent_mask(
            self.horizon
        ).to(x.device)                                      # (horizon, horizon)

        if self.variant == "vanilla":
            out = self.transformer(src, tgt, tgt_mask=causal)  # (B, horizon, D)

        else:  # informer
            # Encoder con ProbSparse + distilling: L=168 → 84 → 42
            memory = self.informer_encoder(src)             # (B, L', D)
            # Decoder vanilla: self-attention causal + cross-attention a memory
            out = self.transformer_decoder(
                tgt, memory, tgt_mask=causal
            )                                               # (B, horizon, D)

        return self.head(out)                               # (B, horizon, n_targets)
