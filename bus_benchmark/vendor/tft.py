# coding=utf-8
# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0
#
# Temporal Fusion Transformer — PyTorch Implementation
# Translated 1:1 from tft_tf_version.py (TensorFlow/Keras).
# Each class/method includes TF source line references for traceability.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GLU — maps to apply_gating_layer(), tft_tf_version.py L107-145
# =============================================================================
class GatedLinearUnit(nn.Module):
    """Applies a Gated Linear Unit (GLU) to an input.

    TF ref: apply_gating_layer() L107-145.
    activation_layer = Dense(hidden, activation)(x)
    gated_layer      = Dense(hidden, sigmoid)(x)
    output           = activation_layer * gated_layer
    """

    def __init__(self, input_size: int, hidden_size: int,
                 dropout_rate: Optional[float] = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.fc_act = nn.Linear(input_size, hidden_size)
        self.fc_gate = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dropout is not None:
            x = self.dropout(x)
        act = self.fc_act(x)                # activation branch
        gate = torch.sigmoid(self.fc_gate(x))  # gating branch
        return act * gate, gate


# =============================================================================
# Add & Norm — maps to add_and_norm(), tft_tf_version.py L148-159
# =============================================================================
class AddAndNorm(nn.Module):
    """Skip connection followed by LayerNorm.  TF ref: L148-159."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)


# =============================================================================
# GRN — maps to gated_residual_network(), tft_tf_version.py L162-224
# =============================================================================
class GatedResidualNetwork(nn.Module):
    """Gated Residual Network.  TF ref: L162-224.

    Flow:  skip ← (proj(x) if output_size else x)
           η  = Dense(x)  [+ Dense_no_bias(context)]
           η  = ELU(η)
           η  = Dense(η)
           GLU_out, gate = GLU(η)
           return LayerNorm(skip + GLU_out),  gate
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: Optional[int] = None,
                 dropout_rate: Optional[float] = None,
                 use_context: bool = False):
        super().__init__()
        self.out_size = output_size if output_size is not None else hidden_size

        # Skip projection (TF L186-192)
        if output_size is not None and output_size != input_size:
            self.skip_proj = nn.Linear(input_size, self.out_size)
        else:
            self.skip_proj = None

        # Feed-forward (TF L195-212)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ctx_proj = (nn.Linear(hidden_size, hidden_size, bias=False)
                         if use_context else None)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Gating + residual (TF L214-224)
        self.glu = GatedLinearUnit(hidden_size, self.out_size, dropout_rate)
        self.add_norm = AddAndNorm(self.out_size)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Skip connection (TF L185-192)
        skip = self.skip_proj(x) if self.skip_proj is not None else x

        # Feed-forward (TF L195-212)
        hidden = self.fc1(x)
        if context is not None and self.ctx_proj is not None:
            hidden = hidden + self.ctx_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        # Gating (TF L214-219)
        gating_out, gate = self.glu(hidden)

        # Add & Norm (TF L222-224)
        out = self.add_norm(gating_out, skip)
        return out, gate


# =============================================================================
# Scaled Dot-Product Attention — tft_tf_version.py L240-275
# =============================================================================
class ScaledDotProductAttention(nn.Module):
    """TF ref: class ScaledDotProductAttention L240-275."""

    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TF L265: temper = sqrt(d_k)
        d_k = k.size(-1)
        temper = math.sqrt(d_k)

        # TF L266-267: attn = batch_dot(q, k, axes=[2,2]) / temper
        attn = torch.bmm(q, k.transpose(1, 2)) / temper  # (B, Tq, Tk)

        # TF L268-271: mask → set masked positions to -inf
        if mask is not None:
            attn = attn.masked_fill(mask == 0.0, float('-inf'))

        attn = F.softmax(attn, dim=-1)  # TF L272
        attn = self.dropout(attn)       # TF L273
        output = torch.bmm(attn, v)     # TF L274
        return output, attn


# =============================================================================
# Interpretable Multi-Head Attention — tft_tf_version.py L278-356
# =============================================================================
class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention with shared value projection.

    TF ref: class InterpretableMultiHeadAttention L278-356.
    Key difference from standard MHA: all heads share the SAME value layer
    (TF L312-317) and heads are averaged (not concatenated) for interpretability.
    """

    def __init__(self, n_head: int, d_model: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head
        self.d_model = d_model

        # Per-head Q/K projections; shared V projection (TF L311-317)
        self.qs_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.ks_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.vs_layer = nn.Linear(d_model, self.d_v, bias=False)  # shared

        self.attention = ScaledDotProductAttention(dropout)
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)  # TF L320
        self.head_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TF ref: __call__ L322-356.

        Args:
            q, k, v: (B, T, d_model)
            mask: (B, T, T)  — causal mask
        Returns:
            output (B, T, d_model), attn_weights (n_head, B, T, T)
        """
        heads, attns = [], []
        vs = self.vs_layer(v)  # shared value — TF L317

        for i in range(self.n_head):
            qs = self.qs_layers[i](q)  # (B, T, d_k)   TF L341
            ks = self.ks_layers[i](k)  # (B, T, d_k)   TF L342
            head, attn = self.attention(qs, ks, vs, mask)  # TF L344
            heads.append(self.head_dropout(head))  # TF L346-347
            attns.append(attn)

        # TF L349-352: stack → mean across heads
        if self.n_head > 1:
            head = torch.stack(heads, dim=0)   # (n_head, B, T, d_v)
            outputs = head.mean(dim=0)          # (B, T, d_v)
        else:
            outputs = heads[0]

        attn = torch.stack(attns, dim=0)  # (n_head, B, T, T)  TF L350

        outputs = self.w_o(outputs)       # TF L353
        outputs = self.out_dropout(outputs)  # TF L354
        return outputs, attn


# =============================================================================
# Static Variable Selection Network — tft_tf_version.py L815-858
# =============================================================================
class StaticVariableSelectionNetwork(nn.Module):
    """Variable selection for static inputs.

    TF ref: static_combine_and_mask() L815-858.
    Input:  (B, num_static, hidden_size)   — pre-embedded static vars
    Output: static_vec (B, hidden_size),  sparse_weights (B, num_static, 1)
    """

    def __init__(self, num_static: int, hidden_size: int,
                 dropout_rate: float):
        super().__init__()
        self.num_static = num_static
        # GRN for variable weights (TF L831-837)
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_static * hidden_size,
            hidden_size=hidden_size,
            output_size=num_static,
            dropout_rate=dropout_rate)
        # Per-variable GRNs (TF L843-849)
        self.per_var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size,
                                 dropout_rate=dropout_rate)
            for _ in range(num_static)])

    def forward(self, embedding: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding: (B, num_static, hidden_size)
        Returns:
            static_vec: (B, hidden_size)
            sparse_weights: (B, num_static, 1)
        """
        B = embedding.size(0)
        # TF L828: flatten
        flat = embedding.reshape(B, -1)
        # TF L831-837: GRN → softmax
        mlp_out, _ = self.flattened_grn(flat)           # (B, num_static)
        sparse_weights = F.softmax(mlp_out, dim=-1)     # TF L839
        sparse_weights = sparse_weights.unsqueeze(-1)   # (B, num_static, 1) TF L840

        # TF L842-849: per-variable GRN
        trans = []
        for i in range(self.num_static):
            e, _ = self.per_var_grns[i](embedding[:, i:i + 1, :])
            trans.append(e)
        transformed = torch.cat(trans, dim=1)  # (B, num_static, hidden) TF L851

        # TF L853-856: weighted sum
        combined = sparse_weights * transformed
        static_vec = combined.sum(dim=1)  # (B, hidden_size) TF L856
        return static_vec, sparse_weights


# =============================================================================
# Temporal Variable Selection Network — tft_tf_version.py L883-931
# =============================================================================
class TemporalVariableSelectionNetwork(nn.Module):
    """Variable selection for temporal (time-varying) inputs.

    TF ref: lstm_combine_and_mask() L883-931.
    Input:  (B, T, hidden_size, num_inputs)  — per-variable embeddings
    Output: temporal_ctx (B, T, hidden_size),
            sparse_weights (B, T, 1, num_inputs)
    """

    def __init__(self, num_inputs: int, hidden_size: int,
                 dropout_rate: float):
        super().__init__()
        self.num_inputs = num_inputs
        # GRN for weight computation (TF L903-910)
        self.flattened_grn = GatedResidualNetwork(
            input_size=hidden_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout_rate=dropout_rate,
            use_context=True)
        # Per-variable GRNs (TF L916-923)
        self.per_var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size,
                                 dropout_rate=dropout_rate)
            for _ in range(num_inputs)])

    def forward(self, embedding: torch.Tensor,
                static_context: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding: (B, T, hidden_size, num_inputs)
            static_context: (B, hidden_size)  — from static VSN
        Returns:
            temporal_ctx: (B, T, hidden_size)
            sparse_weights: (B, T, 1, num_inputs)
        """
        B, T, H, N = embedding.shape
        # TF L896-897: flatten last 2 dims
        flat = embedding.reshape(B, T, H * N)
        # TF L899-900: expand static context
        ctx = static_context.unsqueeze(1)  # (B, 1, hidden)

        # TF L903-910: GRN → softmax
        mlp_out, _ = self.flattened_grn(flat, context=ctx)    # (B, T, N)
        sparse_weights = F.softmax(mlp_out, dim=-1)           # TF L912
        sparse_weights = sparse_weights.unsqueeze(2)          # (B,T,1,N) TF L913

        # TF L916-923: per-variable GRN
        trans = []
        for i in range(N):
            grn_out, _ = self.per_var_grns[i](embedding[..., i])  # (B,T,H)
            trans.append(grn_out)
        transformed = torch.stack(trans, dim=-1)  # (B,T,H,N) TF L925

        # TF L927-929: weighted sum
        combined = sparse_weights * transformed
        temporal_ctx = combined.sum(dim=-1)  # (B, T, H) TF L929
        return temporal_ctx, sparse_weights


# =============================================================================
# TFT — Main model. Maps to _build_base_graph() L779-1029
#        and output projection from build_model() L1031-1098
# =============================================================================
class TFT(nn.Module):
    """Temporal Fusion Transformer core.

    TF ref: TemporalFusionTransformer._build_base_graph() L779-1029
            + build_model() output layer L1043-1045.

    Args:
        num_obs_inputs:   Number of observed (past-only) variables.
        num_known_inputs: Number of known-future variables.
        num_static_inputs: Number of static variables (0 → no static branch).
        hidden_layer_size: Internal state dimension (d_model).
        num_encoder_steps: Encoder (lookback) length.
        num_decoder_steps: Decoder (forecast horizon) length.
        num_heads:  Number of attention heads.
        dropout_rate: Dropout rate.
        num_quantiles: Number of output quantiles (default 3: p10,p50,p90).
        num_stacks: Number of self-attention stacks (default 1).
    """

    def __init__(
        self,
        num_obs_inputs: int,
        num_known_inputs: int,
        num_static_inputs: int,
        hidden_layer_size: int,
        num_encoder_steps: int,
        num_decoder_steps: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_quantiles: int = 3,
        num_stacks: int = 1,
    ):
        super().__init__()
        self.num_obs = num_obs_inputs
        self.num_known = num_known_inputs
        self.num_static = num_static_inputs
        self.hidden = hidden_layer_size
        self.enc_steps = num_encoder_steps
        self.dec_steps = num_decoder_steps
        self.num_heads = num_heads
        self.num_quantiles = num_quantiles
        self.num_stacks = num_stacks
        self.dropout_rate = dropout_rate

        d = hidden_layer_size
        dr = dropout_rate

        # ── 1. Input embeddings (TF get_tft_embeddings L549-553) ──
        # Each continuous variable gets its own Linear(1 → d)
        self.obs_embed = nn.ModuleList(
            [nn.Linear(1, d) for _ in range(num_obs_inputs)])
        self.known_embed = nn.ModuleList(
            [nn.Linear(1, d) for _ in range(num_known_inputs)])

        # ── 2. Static Variable Selection (TF L860) ──
        self.use_static = num_static_inputs > 0
        if self.use_static:
            self.static_vsn = StaticVariableSelectionNetwork(
                num_static_inputs, d, dr)
            # 4 static context GRNs (TF L862-881)
            self.static_ctx_var_sel = GatedResidualNetwork(d, d, dropout_rate=dr)
            self.static_ctx_enrich = GatedResidualNetwork(d, d, dropout_rate=dr)
            self.static_ctx_state_h = GatedResidualNetwork(d, d, dropout_rate=dr)
            self.static_ctx_state_c = GatedResidualNetwork(d, d, dropout_rate=dr)
        else:
            # Learnable fallback when no static inputs provided
            self.static_fallback = nn.Parameter(torch.zeros(1, d))
            nn.init.xavier_uniform_(self.static_fallback.unsqueeze(0))

        # ── 3. Temporal Variable Selection (TF L933-935) ──
        num_hist = num_obs_inputs + num_known_inputs
        num_future = num_known_inputs
        self.hist_vsn = TemporalVariableSelectionNetwork(num_hist, d, dr)
        self.future_vsn = TemporalVariableSelectionNetwork(num_future, d, dr)

        # ── 4. LSTM Encoder / Decoder (TF L938-968) ──
        self.lstm_encoder = nn.LSTM(
            input_size=d, hidden_size=d, batch_first=True)
        self.lstm_decoder = nn.LSTM(
            input_size=d, hidden_size=d, batch_first=True)

        # ── 5. Post-LSTM gated skip (TF L975-977) ──
        self.post_lstm_glu = GatedLinearUnit(d, d, dropout_rate=dr)
        self.post_lstm_norm = AddAndNorm(d)

        # ── 6. Static enrichment (TF L980-987) ──
        self.enrichment_grn = GatedResidualNetwork(
            d, d, dropout_rate=dr, use_context=True)

        # ── 7. Self-attention (TF L990-1003) ──
        self.self_attn_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(num_heads, d, dropout_rate)
            for _ in range(num_stacks)])
        self.post_attn_glus = nn.ModuleList([
            GatedLinearUnit(d, d, dropout_rate=dr) for _ in range(num_stacks)])
        self.post_attn_norms = nn.ModuleList([
            AddAndNorm(d) for _ in range(num_stacks)])

        # ── 8. Position-wise feed-forward / decoder GRN (TF L1006-1010) ──
        self.decoder_grn = GatedResidualNetwork(d, d, dropout_rate=dr)

        # ── 9. Final skip connection (TF L1013-1015) ──
        self.final_glu = GatedLinearUnit(d, d)
        self.final_norm = AddAndNorm(d)

        # ── 10. Output projection (TF build_model L1043-1045) ──
        self.output_proj = nn.Linear(d, num_obs_inputs * num_quantiles)

    def _get_static_contexts(self, static_inputs: Optional[torch.Tensor],
                             B: int) -> Tuple[torch.Tensor, ...]:
        """Compute 4 static context vectors from static VSN.

        TF ref: L860-881.
        Returns: (ctx_var_sel, ctx_enrich, ctx_h, ctx_c, static_weights)
        """
        if self.use_static and static_inputs is not None:
            static_enc, s_weights = self.static_vsn(static_inputs)
            ctx_vs, _ = self.static_ctx_var_sel(static_enc)
            ctx_en, _ = self.static_ctx_enrich(static_enc)
            ctx_h, _ = self.static_ctx_state_h(static_enc)
            ctx_c, _ = self.static_ctx_state_c(static_enc)
        else:
            fallback = self.static_fallback.expand(B, -1)
            ctx_vs = ctx_en = ctx_h = ctx_c = fallback
            s_weights = None
        return ctx_vs, ctx_en, ctx_h, ctx_c, s_weights

    # -----------------------------------------------------------------
    def _embed_inputs(
        self,
        obs_inputs: torch.Tensor,       # (B, T_enc, num_obs)
        known_inputs: torch.Tensor,      # (B, T_total, num_known)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed continuous variables.  TF ref: get_tft_embeddings L549-597.

        Returns:
            hist_emb: (B, T_enc, hidden, num_obs + num_known)
            future_emb: (B, T_dec, hidden, num_known)
        """
        enc = self.enc_steps

        # Embed each observed variable (TF L556-560)
        obs_list = [self.obs_embed[i](obs_inputs[..., i:i + 1])
                    for i in range(self.num_obs)]
        obs_emb = torch.stack(obs_list, dim=-1)  # (B, T_enc, H, num_obs)

        # Embed each known variable (TF L584-588)
        known_list = [self.known_embed[i](known_inputs[..., i:i + 1])
                      for i in range(self.num_known)]
        known_emb = torch.stack(known_list, dim=-1)  # (B, T_total, H, num_known)

        # Historical = concat(known[:enc], obs[:enc]) on var dim (TF L799-810)
        hist_emb = torch.cat(
            [known_emb[:, :enc, :, :], obs_emb], dim=-1)

        # Future = known[enc:] (TF L813)
        future_emb = known_emb[:, enc:, :, :]

        return hist_emb, future_emb

    # -----------------------------------------------------------------
    @staticmethod
    def _get_decoder_mask(x: torch.Tensor) -> torch.Tensor:
        """Causal mask for self-attention.  TF ref: get_decoder_mask L228-237.

        TF logic: cumsum(eye(T), axis=1) → lower-triangular matrix of 1s.
        """
        T = x.size(1)
        # Lower-triangular ones matrix (position i attends to 0..i)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        return mask.unsqueeze(0)  # (1, T, T) — broadcasts over batch

    # -----------------------------------------------------------------
    def forward(
        self,
        obs_inputs: torch.Tensor,            # (B, T_enc, num_obs)
        known_inputs: torch.Tensor,           # (B, T_enc+T_dec, num_known)
        static_inputs: Optional[torch.Tensor] = None,  # (B, num_static, hidden)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full TFT forward pass.

        TF ref: _build_base_graph() L779-1029 + build_model() L1043-1045.

        Returns:
            outputs: (B, T_dec, num_obs * num_quantiles)
            attention_components: dict with interpretability weights
        """
        # ── 1. Embed inputs ──
        hist_emb, future_emb = self._embed_inputs(obs_inputs, known_inputs)

        # ── 2. Static contexts ──
        ctx_vs, ctx_en, ctx_h, ctx_c, static_wts = \
            self._get_static_contexts(static_inputs, obs_inputs.size(0))

        # ── 3. Temporal variable selection (TF L933-935) ──
        hist_feat, hist_flags = self.hist_vsn(hist_emb, ctx_vs)
        future_feat, future_flags = self.future_vsn(future_emb, ctx_vs)

        # ── 4. LSTM (TF L962-970) ──
        # Encoder with static init states
        h0 = ctx_h.unsqueeze(0).contiguous()   # (1, B, d)
        c0 = ctx_c.unsqueeze(0).contiguous()   # (1, B, d)
        
        hist_enc, (h_state, c_state) = self.lstm_encoder(
            hist_feat, (h0, c0))
                
        future_lstm, _ = self.lstm_decoder(
            future_feat, (h_state, c_state))

        # Concat encoder + decoder LSTM outputs (TF L970)
        lstm_layer = torch.cat([hist_enc, future_lstm], dim=1)

        # ── 5. Post-LSTM gated skip (TF L975-977) ──
        input_embed = torch.cat([hist_feat, future_feat], dim=1)
        gated_lstm, _ = self.post_lstm_glu(lstm_layer)
        temporal_feat = self.post_lstm_norm(gated_lstm, input_embed)

        # ── 6. Static enrichment (TF L980-987) ──
        ctx_en_expanded = ctx_en.unsqueeze(1)  # (B, 1, d)
        enriched, _ = self.enrichment_grn(temporal_feat, context=ctx_en_expanded)

        # ── 7. Self-attention with causal mask (TF L990-1003) ──
        mask = self._get_decoder_mask(enriched)
        x = enriched
        for i in range(self.num_stacks):
            attn_out, self_att = self.self_attn_layers[i](x, x, x, mask=mask)
            gated_attn, _ = self.post_attn_glus[i](attn_out)
            x = self.post_attn_norms[i](gated_attn, x)

        # ── 8. Position-wise feed-forward (TF L1006-1010) ──
        decoder_out, _ = self.decoder_grn(x)

        # ── 9. Final skip connection (TF L1013-1015) ──
        final_gated, _ = self.final_glu(decoder_out)
        transformer_layer = self.final_norm(final_gated, temporal_feat)

        # ── 10. Output — decoder portion only (TF L1043-1045) ──
        outputs = self.output_proj(
            transformer_layer[:, self.enc_steps:, :])  # (B, T_dec, obs*Q)

        # Attention components for interpretability (TF L1018-1027)
        attention_components = {
            'decoder_self_attn': self_att,
            'static_flags': static_wts,
            'historical_flags': hist_flags,
            'future_flags': future_flags,
        }
        return outputs, attention_components

    def save_checkpoint(self, path: str):
        """Save the TFT model state and config."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        config = {
            "num_obs_inputs": self.num_obs,
            "num_known_inputs": self.num_known,
            "num_static_inputs": self.num_static,
            "hidden_layer_size": self.hidden,
            "num_encoder_steps": self.enc_steps,
            "num_decoder_steps": self.dec_steps,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "num_quantiles": self.num_quantiles,
            "num_stacks": self.num_stacks,
        }
        torch.save({"config": config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: str, device=None):
        """Load the TFT model state and config."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
