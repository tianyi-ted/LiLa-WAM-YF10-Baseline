"""CPU smoke checks; run from the project root, no DINO weights required."""
import torch
from models.vla_model_fm import VLAModel, fixed_layer_norm


def main():
    torch.manual_seed(0)
    kw = dict(action_dim=14, proprio_dim=16, hidden_dim=32, depth=1,
              num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4,
              adapter_depth=1, task_cond_dim=32, use_global_intent=True,
              inject_global_intent=True, intent_hidden_dim=32,
              intent_pool_hidden_dim=32, intent_pool_grid_size=(3, 4))
    mean = VLAModel(**kw)
    explicit = VLAModel(**kw, intent_pooling='mean')
    explicit.load_state_dict(mean.state_dict(), strict=True)
    assert not any('intent_pool.' in k for k in mean.state_dict())
    patches, future = torch.randn(2, 12, 32), torch.randn(2, 12, 32)
    state, task = torch.randn(2, 1, 16), torch.randn(2, 32)
    a = mean.predict_global_intent(patches, state, task)
    assert torch.equal(a, explicit.predict_global_intent(patches, state, task))
    expected = mean.intent_predictor(fixed_layer_norm(patches.mean(1)), state, task,
                                    torch.full((2,), 32.0))
    assert torch.equal(a, expected)
    attn = VLAModel(**kw, intent_pooling='task_attention')
    pred = attn.predict_global_intent(patches, state, task)
    target = attn.make_global_intent_target(patches, future)
    assert torch.equal(target, mean.make_global_intent_target(patches, future))
    assert not target.requires_grad
    loss = (pred - target).square().mean()
    loss.backward()
    for name in ('query.1.weight', 'patch_proj.1.weight', 'attention.in_proj_weight', 'output.weight'):
        grad = dict(attn.intent_pool.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0, name
    before = attn.intent_pool.query[1].weight.detach().clone()
    torch.optim.AdamW(attn.parameters(), lr=1e-3).step()
    assert not torch.equal(before, attn.intent_pool.query[1].weight)
    print('PASS: mean compatibility, unchanged detached target, attention forward/backward and optimizer step')


if __name__ == '__main__':
    main()
