import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VisualFeatureAdapter(nn.Module):
    """Perceiver-style adapter: learnable queries cross-attend to vision tokens, output (B, num_queries, hidden_dim)."""
    def __init__(self, feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        self.input_norm = nn.LayerNorm(feat_dim)
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features):
        """
        features: (B, N, feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = features.shape[0]
        memory = self.input_norm(features)
        memory = self.feature_proj(memory)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        return out


class MultiLayerConcatFusion(nn.Module):
    """
    Multi-layer DINO feature fusion (concat mode):
      - Each layer is normalized by its own LayerNorm first (feature norms differ
        a lot across DINOv3 layers; shallow layers would be drowned out otherwise)
      - Concatenate token-wise along the feature dim -> (B, N, L*feat_dim)
      - Linear / MLP projection down to -> (B, N, out_dim)
    The output is then fed into a single VisualFeatureAdapter.
    """
    def __init__(self, feat_dim, num_layers, out_dim, proj_type="linear", pre_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        if pre_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported concat proj_type: {proj_type}")

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)] (all layers must share the same N)"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        if self.pre_norm:
            feats_list = [ln(f) for ln, f in zip(self.layer_norms, feats_list)]
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


class FutureFeatureDecoder(nn.Module):
    """
    Future-frame feature prediction decoder (Transformer):
      - Learnable query tokens as Q
      - cond tokens produced by the action head as KV (cross-attention)
      - Output projected to the target feature dim (= DINO hidden_size), supervised
        against future-frame patch features with a cosine loss

    Number of queries = number of future-frame patch tokens (dense prediction,
    one query per patch position).
    """
    def __init__(self, num_queries, hidden_dim, out_dim,
                 num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.out_dim = out_dim

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Normalize cond tokens before they enter cross-attention
        self.kv_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_tokens):
        """
        cond_tokens: (B, M, hidden_dim) — observation tokens evolved by the DiT
        Returns: (B, num_queries, out_dim) predicted future-frame patch features
        """
        B = cond_tokens.shape[0]
        memory = self.kv_norm(cond_tokens)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        out = self.out_proj(self.out_norm(out))
        return out


def future_feature_cosine_loss(pred, target):
    """
    Per-token cosine loss = mean(1 - cos_sim).
    pred / target: (B, M, D). Computed in float32 for numerical stability
    under bf16 training.
    """
    assert pred.shape == target.shape, \
        f"future-feat pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)} " \
        f"(check patch count: num_queries should equal the future-frame DINO patch token count)"
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)        # (B, M)
    return (1.0 - cos).mean()


def fixed_layer_norm(x):
    """Layer-normalize the last dimension without trainable affine parameters."""
    return F.layer_norm(x.float(), (x.shape[-1],)).to(dtype=x.dtype)


def fixed_spatial_pool(patch_features, grid_size, region_grid=(2, 4)):
    """Pool patch tokens into a fixed 2D region grid without learnable matching."""
    if patch_features.ndim != 3:
        raise ValueError(f"expected patch features [B,N,D], got {tuple(patch_features.shape)}")
    grid_h, grid_w = grid_size
    region_h, region_w = region_grid
    B, N, D = patch_features.shape
    if N != grid_h * grid_w:
        raise ValueError(
            f"spatial patch count mismatch: got N={N}, expected {grid_h}x{grid_w}="
            f"{grid_h * grid_w}"
        )
    x = patch_features.transpose(1, 2).reshape(B, D, grid_h, grid_w)
    pooled = F.adaptive_avg_pool2d(x, (region_h, region_w))
    return pooled.permute(0, 2, 3, 1).reshape(B, region_h * region_w, D)


class SpatialIntentPredictor(nn.Module):
    """Predict fixed-region future changes from current causal observations."""
    def __init__(self, vision_dim, proprio_dim, task_cond_dim=None,
                 num_regions=8, hidden_dim=256, horizon_dim=32):
        super().__init__()
        self.num_regions = num_regions
        self.use_task_cond = task_cond_dim is not None
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
        )
        if self.use_task_cond:
            self.task_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
            )
        self.horizon_mlp = nn.Sequential(
            SinusoidalPosEmb(horizon_dim),
            nn.Linear(horizon_dim, hidden_dim),
            nn.SiLU(),
        )
        num_inputs = 4 if self.use_task_cond else 3
        self.predictor = nn.Sequential(
            nn.Linear(num_inputs * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim),
        )

    def forward(self, current_regions, qpos_history, task_cond, horizon):
        B, R, _ = current_regions.shape
        if R != self.num_regions:
            raise ValueError(f"expected {self.num_regions} spatial regions, got {R}")
        state = self.state_proj(qpos_history[:, -1, :]).unsqueeze(1).expand(-1, R, -1)
        parts = [self.vision_proj(current_regions), state]
        if self.use_task_cond:
            if task_cond is None:
                raise ValueError("SpatialIntentPredictor requires task_cond")
            task = self.task_proj(task_cond).unsqueeze(1).expand(-1, R, -1)
            parts.append(task)
        h = self.horizon_mlp(horizon).unsqueeze(1).expand(-1, R, -1)
        parts.append(h)
        return self.predictor(torch.cat(parts, dim=-1))


def spatial_intent_loss(pred, target):
    """Region-wise Smooth-L1 loss for fixed spatial future-change targets."""
    pred_f, target_f = pred.float(), target.float()
    loss = F.smooth_l1_loss(pred_f, target_f)
    return loss, {
        "spatial_target_norm": target_f.norm(dim=-1).mean().detach().item(),
        "spatial_pred_norm": pred_f.norm(dim=-1).mean().detach().item(),
    }


class GlobalIntentPredictor(nn.Module):
    """Predict a global future-change vector from causal, current-time inputs."""
    def __init__(self, vision_dim, proprio_dim, task_cond_dim=None,
                 hidden_dim=512, horizon_dim=32):
        super().__init__()
        self.use_task_cond = task_cond_dim is not None

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
        )
        if self.use_task_cond:
            self.task_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
            )

        self.horizon_mlp = nn.Sequential(
            SinusoidalPosEmb(horizon_dim),
            nn.Linear(horizon_dim, hidden_dim),
            nn.SiLU(),
        )
        num_inputs = 4 if self.use_task_cond else 3
        self.predictor = nn.Sequential(
            nn.Linear(num_inputs * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim),
        )

    def forward(self, current_global, qpos_history, task_cond, horizon):
        # The latest proprioceptive state is the causal state for this chunk.
        current_state = qpos_history[:, -1, :]
        parts = [
            self.vision_proj(current_global),
            self.state_proj(current_state),
        ]
        if self.use_task_cond:
            if task_cond is None:
                raise ValueError("GlobalIntentPredictor requires task_cond")
            parts.append(self.task_proj(task_cond))
        parts.append(self.horizon_mlp(horizon))
        return self.predictor(torch.cat(parts, dim=-1))


def global_intent_loss(pred, target, cosine_weight=0.0, cosine_eps=1.0e-4):
    """Smooth-L1 intent regression with an optional non-static cosine term."""
    pred_f = pred.float()
    target_f = target.float()
    loss_smooth_l1 = F.smooth_l1_loss(pred_f, target_f)
    target_norm = target_f.norm(dim=-1)
    active = target_norm > cosine_eps
    loss_cosine = pred_f.new_zeros(())
    if cosine_weight > 0.0 and active.any():
        cosine = F.cosine_similarity(pred_f[active], target_f[active], dim=-1)
        loss_cosine = (1.0 - cosine).mean()
    loss = loss_smooth_l1 + cosine_weight * loss_cosine
    return loss, {
        "loss_intent_smooth_l1": loss_smooth_l1.detach().item(),
        "loss_intent_cosine": loss_cosine.detach().item(),
        "intent_target_norm": target_norm.mean().detach().item(),
        "intent_pred_norm": pred_f.norm(dim=-1).mean().detach().item(),
        "intent_static_ratio": (~active).float().mean().detach().item(),
    }


class DiTBlock(nn.Module):
    """
    DiT block with Self-Attention + MLP, conditioned via adaLN-zero on time embedding.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, attn_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn1(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class VLAModel(nn.Module):
    """
    VLA model (DINOv3 version, no language input).

    Architecture:
    1. Action / Proprio / Register / multi-layer DINO tokens are concatenated
       and processed jointly by a DiT
    2. Bi-directional self-attention (Flow Matching predicts the whole chunk in parallel)
    3. State injection: placeholder zeros are replaced once at a given layer
       (avoids signal inflation from repeated accumulation)
    4. Multi-layer DINO features: per-layer independent adapters, or a single
       adapter after concat fusion
    5. Optional future-frame feature prediction: an auxiliary decoder predicts
       future DINO patch features from the evolved observation tokens
    """
    def __init__(self,
                 action_dim=14,
                 proprio_dim=16,
                 hidden_dim=512,
                 num_heads=4,
                 depth=12,
                 action_len=32,
                 proprio_len=1,
                 num_registers=2,
                 dino_feat_dims=(1024,),     # feat_dim of each extracted layer; len = number of layers
                 vlm_num_queries=64,         # number of queries per adapter
                 adapter_depth=2,
                 state_inject_start=0,
                 task_cond_dim=None,         # task condition vector dim (DINOv3 CLS difference feature, e.g. 1024); None = disabled
                 # --- Multi-layer feature fusion mode ---
                 fusion_mode="per_layer",    # "per_layer" (independent adapter per layer) | "concat" (concat + project, single adapter)
                 concat_proj_type="linear",  # projection type in concat mode: "linear" | "mlp"
                 concat_pre_norm=True,       # apply per-layer LayerNorm before concat in concat mode
                 concat_out_dim=None,        # projection target dim in concat mode; None -> = dino_feat_dims[0]
                 # --- Future feature prediction ---
                 use_future_feat=False,
                 future_feat_num_queries=None,        # = future-frame patch token count (dense prediction)
                 future_feat_out_dim=None,            # = DINO hidden_size; None -> = dino_feat_dims[0]
                 future_feat_depth=2,
                 future_feat_heads=4,
                 # --- Causal global future intent ---
                 use_global_intent=False,
                 inject_global_intent=False,
                 intent_horizon=32,
                 intent_hidden_dim=512,
                 intent_horizon_dim=32,
                 intent_feat_dim=None,
                 intent_gate_init=0.0,
                 intent_gate_trainable=True,
                 # --- Fixed spatial interaction intent ---
                 use_spatial_intent=False,
                 inject_spatial_intent=False,
                 inject_spatial_residual=False,
                 spatial_grid_size=(15, 20),
                 spatial_region_grid=(2, 4),
                 spatial_hidden_dim=256,
                 spatial_horizon_dim=32,
                 spatial_gate_init=0.1,
                 spatial_gate_trainable=True,
                 spatial_residual_layer_idx=-1,
                 spatial_residual_prefix_tokens=0,
                 spatial_residual_scale=1.0,
                 spatial_residual_injection="before_fusion",
                 spatial_max_gate=None,
                 spatial_warmup_epochs=0,
                 spatial_ramp_epochs=0,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.state_inject_start = state_inject_start
        self.num_dino_layers = len(dino_feat_dims)
        self.use_task_cond = task_cond_dim is not None

        self.fusion_mode = fusion_mode
        assert fusion_mode in ("per_layer", "concat"), f"Unknown fusion_mode: {fusion_mode}"

        self.use_future_feat = use_future_feat
        self.use_global_intent = use_global_intent
        self.inject_global_intent = inject_global_intent
        self.intent_horizon = float(intent_horizon)

        # --- 1. Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- 2. Projections & PosEmb ---
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))

        # Type embeddings: action, proprio + DINO
        # per_layer: one type emb per DINO layer; concat: a single fused DINO token set -> one type emb
        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_proprio = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        num_dino_type_emb = 1 if fusion_mode == "concat" else self.num_dino_layers
        self.type_emb_dino_layers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(num_dino_type_emb)
        ])

        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_proprio, std=0.02)
        for emb in self.type_emb_dino_layers:
            nn.init.normal_(emb, std=0.02)

        # --- 3. Register Tokens ---
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- 3b. Task Condition Token (DINOv3 CLS difference feature) ---
        # Project task_cond_dim -> hidden_dim and append as one extra token
        if self.use_task_cond:
            self.task_cond_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.type_emb_task_cond = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.type_emb_task_cond, std=0.02)

        # --- 3c. Causal Global Intent ---
        # The same modules exist for G-aux and G2 so their parameter counts and
        # initialization are matched; inject_global_intent only controls whether
        # the predicted token is appended to the DiT sequence.
        if self.use_global_intent:
            intent_feat_dim = intent_feat_dim or dino_feat_dims[0]
            self.intent_predictor = GlobalIntentPredictor(
                vision_dim=intent_feat_dim,
                proprio_dim=proprio_dim,
                task_cond_dim=task_cond_dim,
                hidden_dim=intent_hidden_dim,
                horizon_dim=intent_horizon_dim,
            )
            self.intent_to_token = nn.Sequential(
                nn.LayerNorm(intent_feat_dim),
                nn.Linear(intent_feat_dim, hidden_dim),
            )
            self.type_emb_intent = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.type_emb_intent, std=0.02)
            # Store the requested gate in raw space so tanh(intent_gate) starts
            # at the configured effective value. A zero gate is conservative;
            # init01/fixed01 explicitly test whether a stronger condition is
            # needed for the action DiT to use the intent token.
            gate_init = float(intent_gate_init)
            if not (-0.999 <= gate_init <= 0.999):
                raise ValueError("intent_gate_init must be in [-0.999, 0.999]")
            raw_gate_init = math.atanh(gate_init)
            self.intent_gate = nn.Parameter(
                torch.tensor(raw_gate_init, dtype=torch.float32),
                requires_grad=bool(intent_gate_trainable),
            )
            self.intent_gate_init = gate_init
        self.intent_gate_trainable = bool(intent_gate_trainable)

        self.use_spatial_intent = use_spatial_intent
        self.inject_spatial_intent = inject_spatial_intent
        self.inject_spatial_residual = bool(inject_spatial_residual)
        self.spatial_grid_size = tuple(spatial_grid_size)
        self.spatial_region_grid = tuple(spatial_region_grid)
        self.num_spatial_regions = int(self.spatial_region_grid[0] * self.spatial_region_grid[1])
        self.spatial_gate_init = float(spatial_gate_init)
        self.spatial_max_gate = None if spatial_max_gate is None else float(spatial_max_gate)
        self.spatial_schedule_scale = 1.0
        self.spatial_warmup_epochs = int(spatial_warmup_epochs)
        self.spatial_ramp_epochs = int(spatial_ramp_epochs)
        if self.spatial_max_gate is not None and not (0 < self.spatial_max_gate <= 1):
            raise ValueError("spatial_max_gate must be in (0, 1]")
        self.spatial_gate_trainable = bool(spatial_gate_trainable)
        self.spatial_residual_layer_idx = int(spatial_residual_layer_idx)
        self.spatial_residual_prefix_tokens = int(spatial_residual_prefix_tokens)
        self.spatial_residual_scale = float(spatial_residual_scale)
        self.spatial_residual_injection = str(spatial_residual_injection)
        if self.spatial_residual_injection not in ("before_fusion", "after_fusion"):
            raise ValueError(
                "spatial_residual_injection must be 'before_fusion' or 'after_fusion'"
            )
        if self.use_spatial_intent:
            self.spatial_intent_predictor = SpatialIntentPredictor(
                vision_dim=dino_feat_dims[0],
                proprio_dim=proprio_dim,
                task_cond_dim=task_cond_dim,
                num_regions=self.num_spatial_regions,
                hidden_dim=spatial_hidden_dim,
                horizon_dim=spatial_horizon_dim,
            )
            self.spatial_to_tokens = nn.Sequential(
                nn.LayerNorm(dino_feat_dims[0]),
                nn.Linear(dino_feat_dims[0], hidden_dim),
            )
            self.type_emb_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_regions, hidden_dim)
            )
            nn.init.normal_(self.type_emb_spatial, std=0.02)
            if not (-0.999 <= self.spatial_gate_init <= 0.999):
                raise ValueError("spatial_gate_init must be in [-0.999, 0.999]")
            gate_ratio = self.spatial_gate_init if self.spatial_max_gate is None else min(
                0.999, self.spatial_gate_init / self.spatial_max_gate
            )
            self.spatial_gate = nn.Parameter(
                torch.tensor(math.atanh(gate_ratio), dtype=torch.float32),
                requires_grad=self.spatial_gate_trainable,
            )

        # --- 4. DINO adapters ---
        if fusion_mode == "concat":
            # Concat + project -> single adapter
            fusion_out_dim = concat_out_dim if concat_out_dim is not None else dino_feat_dims[0]
            self.concat_fusion = MultiLayerConcatFusion(
                feat_dim=dino_feat_dims[0],
                num_layers=self.num_dino_layers,
                out_dim=fusion_out_dim,
                proj_type=concat_proj_type,
                pre_norm=concat_pre_norm,
            )
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=fusion_out_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
            ])
        else:
            # per_layer: one independent adapter per layer (NOT shared across layers)
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
                for feat_dim in dino_feat_dims
            ])

        # --- 5. Core Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- 6. Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # --- 7. Future Feature Prediction Decoder (optional) ---
        if self.use_future_feat:
            assert future_feat_num_queries is not None, \
                "use_future_feat=True requires future_feat_num_queries (= future-frame patch token count)"
            ff_out_dim = future_feat_out_dim if future_feat_out_dim is not None else dino_feat_dims[0]
            self.future_feat_decoder = FutureFeatureDecoder(
                num_queries=future_feat_num_queries,
                hidden_dim=hidden_dim,
                out_dim=ff_out_dim,
                num_heads=future_feat_heads,
                num_layers=future_feat_depth,
            )

    def make_global_intent_target(self, current_patch_features, future_patch_features):
        """Construct stop-gradient global change target m_H = LN(u_H) - LN(u_0)."""
        current_global = fixed_layer_norm(current_patch_features.mean(dim=1))
        future_global = fixed_layer_norm(future_patch_features.mean(dim=1))
        return (future_global - current_global).detach()

    def keep_control_gates_fp32(self):
        """Keep tiny scalar gates in fp32 while the main model uses bf16.

        A bf16 parameter cannot resolve the small optimizer updates needed by a
        gate initialized near zero. The gate is cast back to the feature dtype
        at each injection site, so this does not change the surrounding model
        precision or tensor shapes.
        """
        for name in ("intent_gate", "spatial_gate"):
            gate = getattr(self, name, None)
            if isinstance(gate, nn.Parameter):
                gate.data = gate.data.float()
        return self

    def effective_spatial_gate(self):
        gate = torch.tanh(self.spatial_gate)
        if self.spatial_max_gate is not None:
            gate = gate * self.spatial_max_gate
        return gate * self.spatial_schedule_scale

    def set_spatial_schedule(self, epoch_number):
        if self.spatial_warmup_epochs <= 0:
            self.spatial_schedule_scale = 1.0
        elif epoch_number <= self.spatial_warmup_epochs:
            self.spatial_schedule_scale = 0.0
        elif self.spatial_ramp_epochs <= 0:
            self.spatial_schedule_scale = 1.0
        else:
            self.spatial_schedule_scale = min(
                1.0,
                float(epoch_number - self.spatial_warmup_epochs)
                / float(self.spatial_ramp_epochs),
            )

    def predict_global_intent(self, current_patch_features, qpos_history,
                              task_cond=None, horizon=None):
        """Predict m_H using current-time information only."""
        if not self.use_global_intent:
            return None
        current_global = fixed_layer_norm(current_patch_features.mean(dim=1))
        batch_size = current_global.shape[0]
        if horizon is None:
            horizon = torch.full(
                (batch_size,), self.intent_horizon,
                device=current_global.device, dtype=current_global.dtype,
            )
        elif not torch.is_tensor(horizon):
            horizon = torch.full(
                (batch_size,), float(horizon),
                device=current_global.device, dtype=current_global.dtype,
            )
        else:
            horizon = horizon.to(device=current_global.device, dtype=current_global.dtype)
            if horizon.ndim == 0:
                horizon = horizon.expand(batch_size)
        return self.intent_predictor(current_global, qpos_history, task_cond, horizon)

    def make_spatial_intent_target(self, current_patch_features, future_patch_features):
        current_regions = fixed_spatial_pool(
            current_patch_features, self.spatial_grid_size, self.spatial_region_grid
        )
        future_regions = fixed_spatial_pool(
            future_patch_features, self.spatial_grid_size, self.spatial_region_grid
        )
        return (
            fixed_layer_norm(future_regions)
            - fixed_layer_norm(current_regions)
        ).detach()

    def predict_spatial_intent(self, current_patch_features, qpos_history,
                               task_cond=None, horizon=None):
        if not self.use_spatial_intent:
            return None
        current_regions = fixed_spatial_pool(
            current_patch_features, self.spatial_grid_size, self.spatial_region_grid
        )
        B = current_regions.shape[0]
        if horizon is None:
            horizon = torch.full(
                (B,), self.intent_horizon,
                device=current_regions.device, dtype=current_regions.dtype,
            )
        elif not torch.is_tensor(horizon):
            horizon = torch.full(
                (B,), float(horizon),
                device=current_regions.device, dtype=current_regions.dtype,
            )
        else:
            horizon = horizon.to(device=current_regions.device, dtype=current_regions.dtype)
            if horizon.ndim == 0:
                horizon = horizon.expand(B)
        return self.spatial_intent_predictor(current_regions, qpos_history, task_cond, horizon)

    def apply_spatial_residual(self, dino_features_list, spatial_intent):
        """Add predicted region changes to patch tokens before the DINO adapter.

        Unlike token appending, this keeps the original visual-token layout and
        directly modifies the spatially corresponding patch conditions.
        """
        if (not self.use_spatial_intent or not self.inject_spatial_residual
                or spatial_intent is None):
            return dino_features_list
        rh, rw = self.spatial_region_grid
        h, w = self.spatial_grid_size
        b, regions, dim = spatial_intent.shape
        if regions != rh * rw:
            raise ValueError(
                f"Expected {rh * rw} spatial regions, got {regions}"
            )
        region_map = spatial_intent.transpose(1, 2).reshape(b, dim, rh, rw)
        patch_delta = F.interpolate(
            region_map, size=(h, w), mode="bilinear", align_corners=False
        ).flatten(2).transpose(1, 2)
        patch_delta = patch_delta * (
            self.effective_spatial_gate().to(patch_delta.dtype)
            * self.spatial_residual_scale
        )
        features = list(dino_features_list)
        layer_idx = self.spatial_residual_layer_idx
        if layer_idx < 0:
            layer_idx += len(features)
        if not (0 <= layer_idx < len(features)):
            raise ValueError(f"Invalid spatial residual layer index: {layer_idx}")
        target = features[layer_idx]
        prefix = self.spatial_residual_prefix_tokens
        if target.shape[1] - prefix != h * w:
            raise ValueError(
                f"Spatial residual expects {h*w} patch tokens after {prefix} "
                f"prefix tokens, got shape {tuple(target.shape)}"
            )
        updated = target.clone()
        updated[:, prefix:, :] = updated[:, prefix:, :] + patch_delta.to(updated.dtype)
        features[layer_idx] = updated
        return features

    def apply_spatial_residual_to_fused(self, fused_features, spatial_intent):
        """Inject spatial residual after multi-layer fusion and its LayerNorms."""
        if (not self.use_spatial_intent or not self.inject_spatial_residual
                or spatial_intent is None):
            return fused_features
        rh, rw = self.spatial_region_grid
        h, w = self.spatial_grid_size
        b, regions, dim = spatial_intent.shape
        prefix = self.spatial_residual_prefix_tokens
        if regions != rh * rw:
            raise ValueError(f"Expected {rh * rw} spatial regions, got {regions}")
        if fused_features.shape[1] - prefix != h * w:
            raise ValueError(
                f"Fused spatial residual expects {h*w} patch tokens after {prefix} "
                f"prefix tokens, got shape {tuple(fused_features.shape)}"
            )
        region_map = spatial_intent.transpose(1, 2).reshape(b, dim, rh, rw)
        patch_delta = F.interpolate(
            region_map, size=(h, w), mode="bilinear", align_corners=False
        ).flatten(2).transpose(1, 2)
        patch_delta = patch_delta * (
            self.effective_spatial_gate().to(patch_delta.dtype)
            * self.spatial_residual_scale
        )
        updated = fused_features.clone()
        updated[:, prefix:, :] = updated[:, prefix:, :] + patch_delta.to(updated.dtype)
        return updated

    def forward(self,
                t,
                noisy_actions,
                qpos_history=None,
                dino_features_list=None,    # List[Tensor(B, N_l, feat_dim_l)], ordered as feat_layers
                task_cond=None,             # (B, task_cond_dim) task condition vector
                global_intent=None,          # (B, D_dino), predicted from causal inputs once per chunk
                spatial_intent=None,         # (B, R, D_dino), predicted fixed-region changes
                ):
        """
        Returns:
            final_pred: (B, action_len, action_dim) predicted velocity field
            cond_tokens: (B, M, hidden_dim) observation tokens evolved by the DiT
                (all tokens after the action tokens); used as conditioning (KV)
                for the future-feature decoder
        """
        t_emb = self.time_mlp(t)

        # 1. Action / Proprio tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action

        x_proprio_real = self.proprio_proj(qpos_history) + \
                         self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                         self.type_emb_proprio

        # State injection: zero placeholder first, replaced once at the given layer
        x_proprio = torch.zeros_like(x_proprio_real)

        tokens_list = [x_action, x_proprio]
        proprio_start = x_action.shape[1]
        proprio_end = proprio_start + x_proprio.shape[1]

        # 2. Register Tokens
        if self.num_registers > 0:
            regs = self.register_tokens.expand(noisy_actions.shape[0], -1, -1)
            tokens_list.append(regs)

        # 2b. Task Condition Token (projected and appended as one extra token)
        if self.use_task_cond:
            assert task_cond is not None, "use_task_cond=True but task_cond was not passed to forward"
            tc = self.task_cond_proj(task_cond).unsqueeze(1)      # (B, 1, hidden_dim)
            tc = tc + self.type_emb_task_cond
            tokens_list.append(tc)

        # 2c. Predicted future intent token. The real future target is never
        # passed here; both training and inference inject only global_intent.
        if self.use_global_intent and self.inject_global_intent:
            if global_intent is None:
                raise ValueError("inject_global_intent=True but global_intent was not provided")
            intent_token = self.intent_to_token(global_intent).unsqueeze(1)
            intent_token = torch.tanh(self.intent_gate).to(global_intent.dtype) * (
                intent_token + self.type_emb_intent
            )
            tokens_list.append(intent_token)

        if self.use_spatial_intent and self.inject_spatial_intent:
            if spatial_intent is None:
                raise ValueError("inject_spatial_intent=True but spatial_intent was not provided")
            spatial_tokens = self.spatial_to_tokens(spatial_intent)
            self.effective_spatial_gate().to(spatial_intent.dtype) * (
                spatial_tokens + self.type_emb_spatial
            )
            tokens_list.append(spatial_tokens)

        if (self.use_spatial_intent and self.inject_spatial_residual
                and self.spatial_residual_injection == "before_fusion"):
            dino_features_list = self.apply_spatial_residual(
                dino_features_list, spatial_intent
            )

        # 3. Multi-layer DINO Feature Tokens
        #    per_layer: independent adapter per layer; concat: concat + project -> single adapter
        if dino_features_list is not None:
            assert len(dino_features_list) == self.num_dino_layers, \
                f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
            if self.fusion_mode == "concat":
                fused = self.concat_fusion(dino_features_list)       # (B, N, fusion_out_dim)
                if (self.use_spatial_intent and self.inject_spatial_residual
                        and self.spatial_residual_injection == "after_fusion"):
                    fused = self.apply_spatial_residual_to_fused(
                        fused, spatial_intent
                    )
                cond = self.dino_adapters[0](fused)                  # (B, Q, hidden_dim)
                tokens_list.append(cond + self.type_emb_dino_layers[0])
            else:
                for layer_idx, feats in enumerate(dino_features_list):
                    cond = self.dino_adapters[layer_idx](feats)
                    tokens_list.append(cond + self.type_emb_dino_layers[layer_idx])

        # 4. Concat all tokens
        x = torch.cat(tokens_list, dim=1)

        # 5. Transformer Blocks
        state_injected = False

        for i, block in enumerate(self.blocks):
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True

            x = block(x, t_emb)

        # 6. Output Head
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)

        # Evolved observation tokens (all tokens after the action tokens),
        # used as cond tokens for future-feature prediction
        cond_tokens = x[:, self.action_len:, :]

        return {
            "final_pred": final_pred,
            "cond_tokens": cond_tokens,
        }


def calc_flow_matching_loss(
    model,
    x1,
    dino_features_list,
    qpos_history,
    task_cond=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    use_velocity_weighting=False,
    alpha=0.2,
    sigma=0.01,
    # Future Feature Prediction
    future_feat_target=None,
    use_future_feat=False,
    lambda_future_feat=0.5,
    # Causal Global Intent
    current_intent_features=None,
    intent_target_features=None,
    use_global_intent=False,
    lambda_intent=0.0,
    intent_cosine_weight=0.0,
    intent_cosine_eps=1.0e-4,
    intent_horizon=None,
    # Fixed spatial interaction intent
    current_spatial_features=None,
    spatial_target_features=None,
    use_spatial_intent=False,
    lambda_spatial_intent=0.0,
    spatial_horizon=None,
):
    """
    Flow Matching Loss (with optional Task Condition + Future-Feature Prediction)
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. Sample noise x0
    x0 = torch.randn_like(x1)

    # 2. Sample timestep t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")

    # 3. Interpolate
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # 4. Target vector field
    target_v = x1 - x0

    # 5. Forward
    global_intent = None
    if use_global_intent:
        if current_intent_features is None or intent_target_features is None:
            raise ValueError("Global intent training requires current and future patch features")
        global_intent = model.predict_global_intent(
            current_intent_features,
            qpos_history=qpos_history,
            task_cond=task_cond,
            horizon=intent_horizon,
        )

    spatial_intent = None
    if use_spatial_intent:
        if current_spatial_features is None or spatial_target_features is None:
            raise ValueError("Spatial intent training requires current and future patch features")
        spatial_intent = model.predict_spatial_intent(
            current_spatial_features,
            qpos_history=qpos_history,
            task_cond=task_cond,
            horizon=spatial_horizon,
        )

    preds = model(t,
                  noisy_actions=x_t,
                  dino_features_list=dino_features_list,
                  task_cond=task_cond,
                  qpos_history=qpos_history,
                  global_intent=global_intent,
                  spatial_intent=spatial_intent)

    pred_v_final = preds["final_pred"]
    cond_tokens = preds["cond_tokens"]

    # ==================== Velocity Weighting (sample-level) ====================
    if use_velocity_weighting:
        velocities = torch.diff(x1[:, :, :-1], dim=1)
        mean_vel = torch.mean(torch.abs(velocities), dim=(1, 2))
        weights = alpha + (1.0 - alpha) * torch.exp(-(mean_vel ** 2) / (2 * sigma ** 2))
    else:
        weights = torch.ones(bs, device=device)
        mean_vel = torch.zeros(bs, device=device)

    # ==================== Final Loss ====================
    loss_final_unreduced = F.mse_loss(pred_v_final, target_v, reduction='none')
    loss_final_per_sample = torch.mean(loss_final_unreduced, dim=(1, 2))
    loss_mse = torch.mean(loss_final_per_sample * weights)

    # ==================== Future Feature Prediction Loss ====================
    loss_future_feat = torch.tensor(0.0, device=device)

    if use_future_feat and future_feat_target is not None:
        # Gradients flow back into the adapter / DiT (auxiliary supervision that
        # shapes the representation); the target comes from the frozen DINO and
        # naturally carries no gradient
        pred_future = model.future_feat_decoder(cond_tokens)        # (B, M, D_dino)
        loss_future_feat = future_feature_cosine_loss(pred_future, future_feat_target)

    # ==================== Causal Global Intent Loss ====================
    loss_intent = torch.tensor(0.0, device=device)
    intent_info = {
        "loss_intent_smooth_l1": 0.0,
        "loss_intent_cosine": 0.0,
        "intent_target_norm": 0.0,
        "intent_pred_norm": 0.0,
        "intent_static_ratio": 0.0,
    }
    if use_global_intent:
        intent_target = model.make_global_intent_target(
            current_intent_features, intent_target_features
        )
        loss_intent, intent_info = global_intent_loss(
            global_intent,
            intent_target,
            cosine_weight=intent_cosine_weight,
            cosine_eps=intent_cosine_eps,
        )

    loss_spatial_intent = torch.tensor(0.0, device=device)
    spatial_info = {
        "spatial_target_norm": 0.0,
        "spatial_pred_norm": 0.0,
    }
    if use_spatial_intent:
        spatial_target = model.make_spatial_intent_target(
            current_spatial_features, spatial_target_features
        )
        loss_spatial_intent, spatial_info = spatial_intent_loss(
            spatial_intent, spatial_target
        )

    # ==================== Total Loss ====================
    loss = (
        loss_mse
        + lambda_future_feat * loss_future_feat
        + lambda_intent * loss_intent
        + lambda_spatial_intent * loss_spatial_intent
    )

    info = {
        "pred_v": pred_v_final,
        "target_v": target_v,
        "mean_chunk_vel": mean_vel.mean().item(),
        "mean_loss_weight": weights.mean().item(),
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_future_feat": loss_future_feat.item() if isinstance(loss_future_feat, torch.Tensor) else loss_future_feat,
        "loss_intent": loss_intent.item() if isinstance(loss_intent, torch.Tensor) else loss_intent,
        "intent_gate": torch.tanh(model.intent_gate).detach().item() if use_global_intent else 0.0,
    }
    info.update(intent_info)
    info["loss_spatial_intent"] = loss_spatial_intent.item()
    info["spatial_gate"] = (
        model.effective_spatial_gate().detach().item()
        if use_spatial_intent else 0.0
    )
    info.update(spatial_info)
    return loss, info
