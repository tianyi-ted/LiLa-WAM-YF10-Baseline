import torch
import torch.nn as nn
import logging
import json
from transformers import AutoModel, AutoConfig

from .vla_model_fm import VLAModel, calc_flow_matching_loss


logger = logging.getLogger(__name__)


class ModelFactory:
    """Initializes the DINOv3 vision encoder and the action prediction model"""

    @staticmethod
    def create_vision_encoder(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        """
        Load a frozen DINOv3 ViT from a local path (offline).

        Returns:
            model: DINOv3 model (eval mode, frozen)
            hidden_size: int, model hidden dimension
            num_register_tokens: int, number of register tokens (usually 4)
            patch_size: int, ViT patch size (usually 16)
        """
        logger.info(f"Loading frozen DINOv3 vision encoder from {checkpoint_path}...")

        # Read config first for metadata
        config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)

        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

        # Freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        hidden_size = getattr(config, "hidden_size", None)
        num_register_tokens = getattr(config, "num_register_tokens", 4)
        patch_size = getattr(config, "patch_size", 16)

        if hidden_size is None:
            # fallback: infer with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
                out = model(pixel_values=dummy)
                hidden_size = out.last_hidden_state.shape[-1]

        logger.info(
            f"DINOv3 loaded: hidden_size={hidden_size}, "
            f"num_register_tokens={num_register_tokens}, patch_size={patch_size}"
        )
        return model, hidden_size, num_register_tokens, patch_size

    @staticmethod
    def create_action_model(config, dino_hidden_size, num_dino_layers, task_cond_dim=None,
                            patch_size=16, num_register_tokens=0):
        """create VLAModel"""
        logger.info("Initializing VLAModel...")

        model_cfg = config.model
        ae_cfg = model_cfg.action_expert
        ve_cfg = model_cfg.vision_encoder

        # All extracted layers share the same feat_dim (DINOv3 hidden_size is uniform across layers)
        dino_feat_dims = tuple([dino_hidden_size] * num_dino_layers)

        # ---- Multi-layer feature fusion mode ----
        fusion_mode = ve_cfg.get('fusion_mode', 'per_layer')
        concat_cfg = ve_cfg.get('concat', {})
        concat_proj_type = concat_cfg.get('proj_type', 'linear') if concat_cfg else 'linear'
        concat_pre_norm = concat_cfg.get('pre_norm', True) if concat_cfg else True
        concat_out_dim = concat_cfg.get('out_dim', None) if concat_cfg else None
        if concat_out_dim is None:
            concat_out_dim = dino_hidden_size   # project down to DINO hidden_size by default

        # ---- Future feature prediction ----
        ff_cfg = model_cfg['future_feat']
        use_future_feat = ff_cfg['enabled']
        future_feat_depth = ff_cfg['num_decoder_layers']
        future_feat_heads = ff_cfg['num_heads']

        # ---- Causal global intent prediction ----
        intent_cfg = model_cfg.get('global_intent', {})
        use_global_intent = bool(intent_cfg.get('enabled', False)) if intent_cfg else False
        inject_global_intent = bool(intent_cfg.get('inject_into_dit', False)) if intent_cfg else False
        intent_horizon = config.dataset.get('future_frame_offset', None)
        if intent_horizon is None:
            intent_horizon = config.common.action_chunk_size
        if inject_global_intent and not use_global_intent:
            raise ValueError("global_intent.inject_into_dit=True requires global_intent.enabled=True")
        if use_global_intent:
            logger.info(
                f"Global Intent ENABLED: inject_into_dit={inject_global_intent}, "
                f"horizon={intent_horizon}, feature_dim={dino_hidden_size}, "
                f"gate_init={intent_cfg.get('gate_init', 0.0)}, "
                f"gate_trainable={intent_cfg.get('gate_trainable', True)}"
            )

        # ---- Fixed spatial interaction intent ----
        spatial_cfg = model_cfg.get('spatial_intent', {})
        use_spatial_intent = bool(spatial_cfg.get('enabled', False)) if spatial_cfg else False
        inject_spatial_intent = bool(spatial_cfg.get('inject_into_dit', False)) if spatial_cfg else False
        inject_spatial_residual = bool(spatial_cfg.get('inject_as_residual', False)) if spatial_cfg else False
        spatial_residual_injection = spatial_cfg.get('residual_injection', 'before_fusion') if spatial_cfg else 'before_fusion'
        if inject_spatial_intent and not use_spatial_intent:
            raise ValueError("spatial_intent.inject_into_dit=True requires spatial_intent.enabled=True")
        region_grid = tuple(spatial_cfg.get('region_grid', [2, 4])) if spatial_cfg else (2, 4)
        if use_spatial_intent:
            logger.info(
                f"Spatial Intent ENABLED: inject_into_dit={inject_spatial_intent}, "
                f"inject_as_residual={inject_spatial_residual}, "
                f"residual_injection={spatial_residual_injection}, "
                f"region_grid={region_grid}, gate_init={spatial_cfg.get('gate_init', 0.1)}, "
                f"gate_trainable={spatial_cfg.get('gate_trainable', True)}"
            )

        # Future-frame patch token count (number of queries for dense prediction); image_size=(W,H)
        img_w, img_h = tuple(config.dataset.image_size)
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        logger.info(f"Fusion mode: {fusion_mode} "
                    f"(concat_out_dim={concat_out_dim}, proj={concat_proj_type}, pre_norm={concat_pre_norm})")
        if use_future_feat:
            logger.info(f"Future-Feat Prediction ENABLED: "
                        f"num_queries(num_patches)={num_patches}, out_dim={dino_hidden_size}")
        future_target_layer = ff_cfg.get('target_layer', -2)
        feat_layers = list(ve_cfg.get('feat_layers', []))
        spatial_layer_idx = feat_layers.index(future_target_layer) if future_target_layer in feat_layers else -1

        model = VLAModel(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            hidden_dim=ae_cfg.hidden_size,
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len,
            depth=ae_cfg.depth,
            num_heads=ae_cfg.num_heads,
            dino_feat_dims=dino_feat_dims,
            vlm_num_queries=ae_cfg.vlm_adapter_num_queries,
            adapter_depth=ae_cfg.adapter_depth,
            num_registers=model_cfg.num_registers,
            state_inject_start=0,
            task_cond_dim=task_cond_dim,
            fusion_mode=fusion_mode,
            concat_proj_type=concat_proj_type,
            concat_pre_norm=concat_pre_norm,
            concat_out_dim=concat_out_dim,
            use_future_feat=use_future_feat,
            future_feat_num_queries=num_patches,
            future_feat_out_dim=dino_hidden_size,
            future_feat_depth=future_feat_depth,
            future_feat_heads=future_feat_heads,
            use_global_intent=use_global_intent,
            inject_global_intent=inject_global_intent,
            intent_horizon=intent_horizon,
            intent_hidden_dim=intent_cfg.get('hidden_size', ae_cfg.hidden_size) if intent_cfg else ae_cfg.hidden_size,
            intent_horizon_dim=intent_cfg.get('horizon_embed_dim', 32) if intent_cfg else 32,
            intent_feat_dim=dino_hidden_size,
            intent_gate_init=intent_cfg.get('gate_init', 0.0) if intent_cfg else 0.0,
            intent_gate_trainable=intent_cfg.get('gate_trainable', True) if intent_cfg else True,
            use_spatial_intent=use_spatial_intent,
            inject_spatial_intent=inject_spatial_intent,
            inject_spatial_residual=inject_spatial_residual,
            spatial_grid_size=(img_h // patch_size, img_w // patch_size),
            spatial_region_grid=region_grid,
            spatial_hidden_dim=spatial_cfg.get('hidden_size', 256) if spatial_cfg else 256,
            spatial_horizon_dim=spatial_cfg.get('horizon_embed_dim', 32) if spatial_cfg else 32,
            spatial_gate_init=spatial_cfg.get('gate_init', 0.1) if spatial_cfg else 0.1,
            spatial_gate_trainable=spatial_cfg.get('gate_trainable', True) if spatial_cfg else True,
            spatial_residual_layer_idx=spatial_layer_idx,
            spatial_residual_prefix_tokens=(1 + num_register_tokens),
            spatial_residual_scale=spatial_cfg.get('residual_scale', 1.0) if spatial_cfg else 1.0,
            spatial_residual_injection=spatial_residual_injection,
            spatial_max_gate=spatial_cfg.get('max_gate', None) if spatial_cfg else None,
            spatial_warmup_epochs=spatial_cfg.get('warmup_epochs', 0) if spatial_cfg else 0,
            spatial_ramp_epochs=spatial_cfg.get('ramp_epochs', 0) if spatial_cfg else 0,
        )
        return model


class VLAWrapper(nn.Module):
    """
    VLA wrapper (DINOv3 version):
    1. Run DINOv3 on input images with output_hidden_states=True
    2. Extract multi-layer hidden states according to feat_layers, one adapter per layer
    3. Call action_model + flow matching loss
    """
    def __init__(self,
                 vision_encoder,
                 action_model,
                 time_sampler,
                 feat_layers,
                 include_cls_register,
                 num_register_tokens,
                 device,
                 dtype,
                 norm_stats_path,
                 train_config=None,
                 future_feat_target_layer=-1,
                 ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.feat_layers = list(feat_layers)
        self.include_cls_register = include_cls_register
        self.num_register_tokens = num_register_tokens
        self.future_feat_target_layer = future_feat_target_layer

        self.device = device
        self.dtype = dtype

        # Training-related parameters (only used during training)
        self.train_config = train_config
        if self.train_config is not None:
            self.time_mu = train_config['time_mu']
            self.time_sigma = train_config['time_sigma']
            self.use_vel_weight = train_config['use_vel_weight']
            self.vel_weight_alpha = train_config['vel_weight_alpha']
            self.vel_weight_sigma = train_config['vel_weight_sigma']
            self.use_future_feat = train_config.get('use_future_feat', False)
            self.lambda_future_feat = train_config.get('lambda_future_feat', 0.0)
            self.use_global_intent = train_config.get('use_global_intent', False)
            self.lambda_intent = train_config.get('lambda_intent', 0.0)
            self.intent_cosine_weight = train_config.get('intent_cosine_weight', 0.0)
            self.intent_cosine_eps = train_config.get('intent_cosine_eps', 1.0e-4)
            self.intent_horizon = train_config.get('intent_horizon', action_model.intent_horizon)
            self.use_spatial_intent = train_config.get('use_spatial_intent', False)
            self.lambda_spatial_intent = train_config.get('lambda_spatial_intent', 0.0)
            self.spatial_horizon = train_config.get('spatial_horizon', action_model.intent_horizon)
        else:
            # Inference-mode defaults
            self.time_mu = 0.0
            self.time_sigma = 1.0
            self.use_vel_weight = False
            self.vel_weight_alpha = 0.2
            self.vel_weight_sigma = 0.01
            self.use_future_feat = False
            self.lambda_future_feat = 0.0
            self.use_global_intent = action_model.use_global_intent
            self.lambda_intent = 0.0
            self.intent_cosine_weight = 0.0
            self.intent_cosine_eps = 1.0e-4
            self.intent_horizon = action_model.intent_horizon
            self.use_spatial_intent = action_model.use_spatial_intent
            self.lambda_spatial_intent = 0.0
            self.spatial_horizon = action_model.intent_horizon

        logger.info(f"VLAWrapper initialized. feat_layers={self.feat_layers}, "
                    f"include_cls_register={self.include_cls_register}")

        # Load normalization stats
        self.load_norm_stats(norm_stats_path)

    def load_norm_stats(self, path):
        """Read JSON and load action / state min/max for normalization"""
        logger.info(f"Loading normalization stats from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        stats = data['robotwin2']
        action_stats = stats['action']
        state_stats = stats['state']

        act_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        act_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        expected_action = self.action_model.action_proj.in_features
        if act_min.shape != (expected_action,) or act_max.shape != (expected_action,):
            raise ValueError(
                f"Action normalization stats must be {expected_action}D, "
                f"got {tuple(act_min.shape)}/{tuple(act_max.shape)}"
            )
        self.register_buffer('action_min', act_min)
        self.register_buffer('action_max', act_max)
        logger.info(f"Loaded Action stats - Dim: {len(action_stats['min'])}")

        state_min = torch.tensor(state_stats['min'], dtype=torch.float32)
        state_max = torch.tensor(state_stats['max'], dtype=torch.float32)
        expected_state = self.action_model.proprio_proj.in_features
        if state_min.shape != (expected_state,) or state_max.shape != (expected_state,):
            raise ValueError(
                f"State normalization stats must be {expected_state}D, "
                f"got {tuple(state_min.shape)}/{tuple(state_max.shape)}"
            )
        self.register_buffer('state_min', state_min)
        self.register_buffer('state_max', state_max)
        logger.info(f"Loaded State stats - Dim: {len(state_stats['min'])}")

    @torch.no_grad()
    def get_vision_features(self, pixel_values):
        """
        Extract DINOv3 multi-layer hidden states.

        Args:
            pixel_values: (B, 3, H, W), ImageNet normalized

        Returns:
            List[Tensor(B, N, hidden_size)] with length = len(feat_layers), ordered as feat_layers.
            If include_cls_register=True, N = 1 + num_register_tokens + num_patches
            otherwise N = num_patches (the first 1 + num_register_tokens tokens are dropped)
        """
        pixel_values = pixel_values.to(self.device, self.dtype)
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states is a tuple of length num_hidden_layers + 1:
        # [0] is the embedding output, [1..num_layers] are the transformer block outputs
        # feat_layer = -1 -> last layer
        hidden_states = outputs.hidden_states

        feats_list = []
        for layer_idx in self.feat_layers:
            h = hidden_states[layer_idx]   # (B, 1+R+P, D)
            if not self.include_cls_register:
                skip = 1 + self.num_register_tokens
                h = h[:, skip:, :]
            feats_list.append(h)
        return feats_list

    @torch.no_grad()
    def get_future_target_features(self, future_pixel_values):
        """
        Extract future-frame DINO patch token features as the supervision target
        for future-feature prediction (frozen, no gradient).

        Args:
            future_pixel_values: (B, 3, H, W), ImageNet normalized

        Returns:
            Tensor(B, P, hidden_size): patch tokens only (1 CLS + R registers dropped),
            taken from layer future_feat_target_layer
        """
        future_pixel_values = future_pixel_values.to(self.device, self.dtype)
        outputs = self.vision_encoder(
            pixel_values=future_pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        h = outputs.hidden_states[self.future_feat_target_layer]   # (B, 1+R+P, D)
        skip = 1 + self.num_register_tokens
        h = h[:, skip:, :]                                          # (B, P, D) patch tokens only
        return h

    def get_current_intent_features(self, dino_features_list):
        """Return current-frame patch tokens from the same layer as the intent target."""
        if self.future_feat_target_layer not in self.feat_layers:
            raise ValueError(
                f"Global intent target_layer={self.future_feat_target_layer} must also appear "
                f"in vision_encoder.feat_layers={self.feat_layers} to reuse the current DINO forward"
            )
        layer_pos = self.feat_layers.index(self.future_feat_target_layer)
        h = dino_features_list[layer_pos]
        if self.include_cls_register:
            h = h[:, 1 + self.num_register_tokens:, :]
        return h

    def get_current_spatial_features(self, dino_features_list):
        """Return current patch tokens used by the fixed spatial target."""
        return self.get_current_intent_features(dino_features_list)

    def _normalize_tensor(self, x, min_val, max_val):
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)

        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0

        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def denormalize_action(self, norm_action):
        action_min = self.action_min.to(device=norm_action.device, dtype=norm_action.dtype)
        action_max = self.action_max.to(device=norm_action.device, dtype=norm_action.dtype)

        denominator = action_max - action_min
        denominator[denominator < 1e-6] = 1.0

        action = (norm_action + 1) / 2 * denominator + action_min
        return action

    def forward(self, batch):
        """Forward pass and loss computation"""
        # 1. Vision features (multi-layer DINO)
        pixel_values = batch['pixel_values']      # (B, 3, H, W)
        dino_features_list = self.get_vision_features(pixel_values)

        # 2. Action / State preparation
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        qpos_raw = batch['state'].to(self.device, self.dtype)

        if qpos_raw.dim() == 2:
            qpos_raw = qpos_raw.unsqueeze(1)

        # 3. Normalize
        x1 = self.normalize_action(x1_raw)
        qpos = self.normalize_state(qpos_raw)
        # Align state sequence with proprio_len (history_len=1 when state_indices=[0], qpos_history=qpos)
        qpos_history = qpos

        # 4. Task condition vector
        task_cond = None
        if batch.get('task_cond') is not None:
            task_cond = batch['task_cond'].to(self.device, self.dtype)

        # 4b. Future feature target (future-frame patch features from the frozen DINO)
        future_feat_target = None
        if (self.use_future_feat or self.use_global_intent) and batch.get('future_pixel_values') is not None:
            future_feat_target = self.get_future_target_features(batch['future_pixel_values'])

        current_intent_features = None
        if self.use_global_intent:
            current_intent_features = self.get_current_intent_features(dino_features_list)
            if future_feat_target is None:
                raise ValueError("Global intent training requires future_pixel_values")

        current_spatial_features = None
        if self.use_spatial_intent:
            current_spatial_features = self.get_current_spatial_features(dino_features_list)
            if future_feat_target is None:
                raise ValueError("Spatial intent training requires future_pixel_values")

        # 5. Flow Matching Loss
        loss, info_dic = calc_flow_matching_loss(
            self.action_model,
            x1=x1,
            dino_features_list=dino_features_list,
            qpos_history=qpos_history,
            task_cond=task_cond,
            time_sampler=self.time_sampler,
            time_mu=self.time_mu,
            time_sigma=self.time_sigma,
            use_velocity_weighting=self.use_vel_weight,
            alpha=self.vel_weight_alpha,
            sigma=self.vel_weight_sigma,
            future_feat_target=future_feat_target,
            use_future_feat=self.use_future_feat,
            lambda_future_feat=self.lambda_future_feat,
            current_intent_features=current_intent_features,
            intent_target_features=future_feat_target,
            use_global_intent=self.use_global_intent,
            lambda_intent=self.lambda_intent,
            intent_cosine_weight=self.intent_cosine_weight,
            intent_cosine_eps=self.intent_cosine_eps,
            intent_horizon=self.intent_horizon,
            current_spatial_features=current_spatial_features,
            spatial_target_features=future_feat_target,
            use_spatial_intent=self.use_spatial_intent,
            lambda_spatial_intent=self.lambda_spatial_intent,
            spatial_horizon=self.spatial_horizon,
        )

        return loss, info_dic
