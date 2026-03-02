import torch
import torch.nn as nn


class AdaptorModule(nn.Module):
    """
    Adapts La-Proteina representations to Boltz1 confidence head input space.
    
    Supports two modes:
      - "trunk": Uses trunk seqs + local_latents (Option A baseline)
      - "hybrid": Fuses trunk seqs with decoder seqs, plus local_latents (Option C)
    
    In both modes:
      - single_proj input dim = 768 + latent_dim (identical!)
      - pair_proj input dim = 256 (identical!)
      - Only decoder_fusion is new in "hybrid" mode
    """
    
    def __init__(
        self,
        source_mode: str = "trunk",  # "trunk" or "hybrid"
        trunk_dim: int = 768,
        pair_dim: int = 256,
        latent_dim: int = 8,
        target_s_dim: int = 384,
        target_z_dim: int = 128,
    ):
        super().__init__()
        self.source_mode = source_mode
        
        # --- Decoder fusion gate (Option C only) ---
        if source_mode == "hybrid":
            self.decoder_fusion = nn.Sequential(
                nn.LayerNorm(trunk_dim),
                nn.Linear(trunk_dim, trunk_dim, bias=False),
            )
            # Zero-initialize so that at init, hybrid == trunk behavior
            nn.init.zeros_(self.decoder_fusion[1].weight)
        
        # --- Single adaptor: seqs (possibly fused) + local_latents -> s ---
        single_input_dim = trunk_dim + latent_dim  # 768 + 8 = 776
        self.single_proj = nn.Sequential(
            nn.LayerNorm(single_input_dim),
            nn.Linear(single_input_dim, target_s_dim, bias=False),
        )
        
        # --- Pair adaptor: trunk pair_rep -> z ---
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, target_z_dim, bias=False),
        )
    
    def forward(
        self,
        trunk_seqs,       # [b, n, 768]
        trunk_pair,       # [b, n, n, 256]
        local_latents,    # [b, n, 8]
        decoder_seqs=None, # [b, n, 768] -- only in hybrid mode
    ):
        # --- Single representation ---
        if self.source_mode == "hybrid" and decoder_seqs is not None:
            # Fuse decoder signal into trunk seqs via gated addition
            fused_seqs = trunk_seqs + self.decoder_fusion(decoder_seqs)
        else:
            fused_seqs = trunk_seqs
        
        single_in = torch.cat([fused_seqs, local_latents], dim=-1)  # [b, n, 776]
        s = self.single_proj(single_in)                              # [b, n, 384]
        
        # --- Pair representation ---
        z = self.pair_proj(trunk_pair)                               # [b, n, n, 128]
        
        return s, z