"""Opt-in action-shaped future posterior. Does not change legacy VLAModel."""
import torch
from torch import nn
from torch.nn import functional as F

from .vla_model_fm import get_1d_sincos_pos_embed, SinusoidalPosEmb, fixed_layer_norm


class FutureDrivenPosterior(nn.Module):
    """Future patch queries read current context, then one learned summary slot.

    Actions and FM noise/time are deliberately absent from this interface.
    This module is a training-only teacher, not an online policy.
    """
    def __init__(self, vision_dim, state_dim, task_dim, grid_size,
                 hidden_dim=256, num_heads=4, horizon_dim=32):
        super().__init__()
        if hidden_dim % 4 or hidden_dim % num_heads:
            raise ValueError('posterior width must be divisible by 4 and heads')
        self.grid_size = tuple(grid_size)
        self.visual = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim))
        self.state = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim))
        self.task = nn.Sequential(nn.LayerNorm(task_dim), nn.Linear(task_dim, hidden_dim))
        self.horizon = nn.Sequential(SinusoidalPosEmb(horizon_dim), nn.Linear(horizon_dim, hidden_dim), nn.SiLU())
        h, w = self.grid_size
        y = get_1d_sincos_pos_embed(hidden_dim // 2, h)[0]
        x = get_1d_sincos_pos_embed(hidden_dim // 2, w)[0]
        pos = torch.cat((y[:, None].expand(h, w, -1), x[None].expand(h, w, -1)), -1)
        self.register_buffer('position', pos.reshape(1, h*w, hidden_dim), persistent=False)
        self.current_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.future_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.summary_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.summary_norm = nn.LayerNorm(hidden_dim)
        self.summary = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vision_dim))

    def forward(self, current, future, state, task, horizon):
        if current.shape != future.shape or current.shape[1] != self.position.shape[1]:
            raise ValueError('posterior requires matching current/future patch grids')
        if task is None:
            raise ValueError('posterior requires VTT')
        b = current.shape[0]
        horizon = torch.full((b,), float(horizon), device=current.device, dtype=current.dtype)
        pos = self.position.to(current)
        context = torch.cat((self.visual(current) + pos + self.current_type,
                             self.state(state[:, -1]).unsqueeze(1),
                             self.task(task).unsqueeze(1)), dim=1)
        # Future content remains in the residual even when attention reads only current context.
        u = self.visual(future) + pos + self.future_type + self.horizon(horizon).unsqueeze(1)
        memory = self.context_norm(context)
        u = u + self.cross(self.query_norm(u), memory, memory, need_weights=False)[0]
        u = u + self.ff(self.ff_norm(u))
        memory = self.summary_norm(u)
        summary = self.summary(self.summary_query.expand(b, -1, -1), memory, memory, need_weights=False)[0]
        return self.output(summary[:, 0])


def freeze_module(module):
    module.requires_grad_(False)
    module.eval()
    return module


def planwam_flow_loss(action, current, future, features, state, task, actions,
                     stage, posterior=None, teacher=None, horizon=8,
                     lambda_kd=0.1, time_sampler='logit_normal', time_mu=0., time_sigma=1.,
                     noise=None, flow_time=None):
    """Standard Gaussian FM + optional normalized latent distillation.

    Separate function preserves old action-prior/mean/attention-target losses.
    noise and flow_time allow controlled offline teacher intervention tests.
    """
    target = None
    if stage == 'teacher':
        if posterior is None or future is None:
            raise ValueError('teacher stage requires posterior and future features')
        latent = posterior(current, future, state, task, horizon)
    elif stage in ('student', 'action_only'):
        # Deliberately do not pass future to the causal student.
        latent = action.predict_global_intent(current, state, task, horizon=horizon)
        if stage == 'student':
            if teacher is None or future is None:
                raise ValueError('student stage requires frozen teacher and future target')
            if teacher.training or any(p.requires_grad for p in teacher.parameters()):
                raise ValueError('teacher must be frozen and in eval mode')
            with torch.no_grad():
                target = fixed_layer_norm(teacher(current, future, state, task, horizon).float())
    else:
        raise ValueError(f'unknown planwam stage: {stage}')
    x0 = torch.randn_like(actions) if noise is None else noise
    b = actions.shape[0]
    if flow_time is not None:
        t = flow_time
    elif time_sampler == 'logit_normal':
        t = torch.sigmoid(torch.randn(b, device=actions.device) * time_sigma + time_mu)
    elif time_sampler == 'uniform':
        t = torch.rand(b, device=actions.device)
    else:
        raise ValueError(f'unsupported time_sampler: {time_sampler}')
    xt = (1-t[:, None, None]) * x0 + t[:, None, None] * actions
    pred = action(t, noisy_actions=xt, dino_features_list=features,
                  qpos_history=state, task_cond=task, global_intent=latent)['final_pred']
    loss_action = F.mse_loss(pred.float(), (actions - x0).float())
    normalized = fixed_layer_norm(latent.float())
    loss_kd = F.mse_loss(normalized, target) if target is not None else loss_action.new_zeros(())
    total = loss_action + lambda_kd * loss_kd
    info = dict(loss_action=loss_action.detach().item(), loss_kd=loss_kd.detach().item(),
                latent_norm=latent.float().norm(dim=-1).mean().detach().item(),
                latent_std=normalized.std(dim=0, unbiased=False).mean().detach().item(),
                target_norm=target.norm(dim=-1).mean().item() if target is not None else 0.)
    return total, info
