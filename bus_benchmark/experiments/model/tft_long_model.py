"""TFT Model Wrapper for Long Format (Entity-based).

Input: (B, T_enc + T_dec, n_features) from TFTLongDataset.
The last feature dimension is reserved for link_idx.
num_obs_inputs is configured to 2 (travel_time + is_imputed mask).
Creates proper static embeddings for each link independently.
"""

import os
import torch
import torch.nn as nn
import numpy as np

from bus_benchmark.vendor.tft import TFT

class TFTLongModel(nn.Module):
    def __init__(
        self,
        encoder_len: int,
        decoder_len: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_quantiles: int = 3,
        num_stacks: int = 1,
        link_metadata: dict | None = None,
        use_is_imputed: bool = True,
        use_link_id: bool = True,
        use_link_length: bool = True,
        use_tod: bool = True,
        use_dow: bool = True,
        use_stop_sequence: bool = False,
        use_pattern_profile: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_quantiles = num_quantiles
        self.num_stacks = num_stacks
        self.link_metadata = link_metadata
        
        self.use_is_imputed = use_is_imputed
        self.use_link_id = use_link_id
        self.use_link_length = use_link_length
        
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.use_stop_sequence = use_stop_sequence
        self.use_pattern_profile = use_pattern_profile
        
        # Calculate num_known
        self.num_known = (2 if use_tod else 0) + (2 if use_dow else 0) + (1 if use_pattern_profile else 0)
        self.num_obs = 2 if self.use_is_imputed else 1  # Includes actual observation + optional is_imputed mask
        
        num_static = 0
        if link_metadata and "link_ids" in link_metadata:
            num_links_meta = link_metadata["num_links"]
            
            if self.use_link_id:
                self.link_emb = nn.Embedding(num_links_meta, hidden_size)
                num_static += 1
            if self.use_link_length:
                self.length_proj = nn.Linear(1, hidden_size)
                num_static += 1
            if self.use_stop_sequence and "stop_sequence_norms" in link_metadata:
                self.stop_sequence_proj = nn.Linear(1, hidden_size)
                num_static += 1
                
            self.register_buffer("meta_link_ids", torch.from_numpy(link_metadata["link_ids"]).long())
            self.register_buffer("meta_link_lengths", torch.from_numpy(link_metadata["link_lengths"]).float())
            if "stop_sequence_norms" in link_metadata:
                self.register_buffer("meta_stop_sequences", torch.from_numpy(link_metadata["stop_sequence_norms"]).float())
                
            self.has_static = (num_static > 0)
        else:
            self.has_static = False
            
        self.tft = TFT(
            num_obs_inputs=self.num_obs,
            num_known_inputs=self.num_known,
            num_static_inputs=num_static,
            hidden_layer_size=hidden_size,
            num_encoder_steps=encoder_len,
            num_decoder_steps=decoder_len,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_quantiles=num_quantiles,
            num_stacks=num_stacks,
        )
        
    def _build_static(self, link_idx_batch: torch.Tensor) -> torch.Tensor | None:
        """Create static embedding tensor for specific links -> (B, num_static, hidden_size)."""
        if not self.has_static:
            return None
            
        static_list = []
        
        if self.use_link_id:
            b_link_ids = self.meta_link_ids[link_idx_batch]
            l_emb = self.link_emb(b_link_ids) 
            static_list.append(l_emb)
            
        if self.use_link_length:
            b_lengths = self.meta_link_lengths[link_idx_batch].unsqueeze(-1)
            len_proj = self.length_proj(b_lengths)
            static_list.append(len_proj)
            
        if self.use_stop_sequence:
            b_stop_seq = self.meta_stop_sequences[link_idx_batch].unsqueeze(-1)
            ss_proj = self.stop_sequence_proj(b_stop_seq)
            static_list.append(ss_proj)
            
        if not static_list:
            return None
            
        static = torch.stack(static_list, dim=1) # (B, num_static, hidden_size)
        return static

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: (B, T_enc + T_dec, n_features) from TFTLongDataset.
        Returns:
            training: (B, T_dec, 1 * num_quantiles)
            eval:     (B, T_dec, 1)  (p50 extraction)
        """
        B = x.size(0)
        enc = self.encoder_len
        
        # Link index is appended as the last feature in the packing logic
        link_idx_batch = x[:, 0, -1].long()
        
        obs_inputs = x[:, :enc, :self.num_obs]
        known_inputs = x[:, :, self.num_obs:-1]
        
        static = self._build_static(link_idx_batch)
        
        outputs, _attn = self.tft(obs_inputs, known_inputs, static)
        
        Q = self.num_quantiles
        # outputs shape is (B, T_dec, num_obs * Q)
        outputs = outputs.view(B, self.decoder_len, self.num_obs, Q)
        # Extract predictions only for the main target (travel_time, index 0)
        outputs = outputs[:, :, 0, :] # -> (B, T_dec, Q)
        
        if not self.training:
            # Output layout: (B, T_dec, Q)
            reshaped = outputs.view(B, self.decoder_len, 1, Q)
            outputs = reshaped[:, :, :, 1] # Extract p50 -> (B, T_dec, 1)
            
        return outputs

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        config = {
            "encoder_len": self.encoder_len,
            "decoder_len": self.decoder_len,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "num_quantiles": self.num_quantiles,
            "num_stacks": self.num_stacks,
            "use_is_imputed": self.use_is_imputed,
            "use_link_id": self.use_link_id,
            "use_link_length": self.use_link_length,
            "use_tod": self.use_tod,
            "use_dow": self.use_dow,
            "use_stop_sequence": self.use_stop_sequence,
            "use_pattern_profile": self.use_pattern_profile,
        }
        torch.save({
            "config": config,
            "state_dict": self.state_dict(),
            "link_metadata": self.link_metadata
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device=None):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        # Filter deprecated configs to stay forward-compatible with older checkpoints
        deprecated_keys = {"use_route_id", "use_from_stop_id", "use_to_stop_id", "use_lag_1_week", "use_ddn_encoder"}
        config = {k: v for k, v in config.items() if k not in deprecated_keys}
        link_meta = checkpoint.get("link_metadata", None)
        model = cls(**config, link_metadata=link_meta)
        model.load_state_dict(checkpoint["state_dict"])
        return model
