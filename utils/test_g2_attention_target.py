"""CPU regression: PYTHONPATH=. python utils/test_g2_attention_target.py."""
import torch
from models.vla_model_fm import VLAModel, calc_flow_matching_loss, fixed_layer_norm


def main():
    torch.manual_seed(12)
    kw = dict(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1,
              action_len=8, num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4,
              adapter_depth=1, task_cond_dim=32, use_global_intent=True,
              inject_global_intent=True, intent_hidden_dim=32, intent_use_gate=False,
              intent_pooling='task_attention', intent_pool_hidden_dim=32,
              intent_pool_grid_size=(3, 4), inject_task_cond=False)
    old = VLAModel(**kw)
    explicit = VLAModel(**kw, intent_target_pooling='mean')
    new = VLAModel(**kw, intent_target_pooling='current_attention')
    explicit.load_state_dict(old.state_dict(), strict=True)
    new.load_state_dict(old.state_dict(), strict=True)
    patches = torch.randn(2, 12, 32, requires_grad=True)
    future = torch.randn(2, 12, 32, requires_grad=True)
    state, task = torch.randn(2, 1, 4), torch.randn(2, 32)
    a = old.predict_global_intent(patches, state, task)
    assert torch.equal(a, explicit.predict_global_intent(patches, state, task))
    pred, alpha = new.predict_global_intent(patches, state, task, return_attention=True)
    assert alpha.shape == (2, 12)
    assert torch.allclose(alpha.sum(1), torch.ones(2))
    # Future data cannot affect the predictor or selector: neither is an input.
    assert torch.equal(pred, new.predict_global_intent(patches, state, task))
    assert torch.allclose(a, pred, atol=1e-6, rtol=1e-5)
    target = new.make_global_intent_target(patches, future, alpha)
    weights = (0.5 / 12 + 0.5 * alpha.detach()).unsqueeze(-1)
    expected = fixed_layer_norm((weights * future.detach()).sum(1)) - fixed_layer_norm((weights * patches.detach()).sum(1))
    assert torch.allclose(target, expected)
    assert not target.requires_grad
    new.intent_target_attention_mix = 0.0
    assert torch.allclose(new.make_global_intent_target(patches, future, alpha),
                          old.make_global_intent_target(patches, future), atol=2e-6)
    new.intent_target_attention_mix = 1.0
    onehot = torch.zeros(2, 12); onehot[:, 3] = 1
    assert torch.allclose(new.make_global_intent_target(patches, future, onehot),
                          fixed_layer_norm(future.detach()[:, 3]) - fixed_layer_norm(patches.detach()[:, 3]))
    new.intent_target_attention_mix = 0.5
    counter = []
    hook = new.intent_pool.register_forward_hook(lambda *args: counter.append(1))
    loss, info = calc_flow_matching_loss(
        new, x1=torch.randn(2, 8, 4), dino_features_list=[patches],
        qpos_history=state, task_cond=task, current_intent_features=patches,
        intent_target_features=future, use_global_intent=True, lambda_intent=0.1,
    )
    hook.remove()
    assert len(counter) == 1, 'selector must run once per chunk'
    assert torch.isfinite(loss)
    loss.backward()
    assert future.grad is None, 'future target must have no gradient'
    grad = new.intent_pool.query[1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    before = new.intent_pool.query[1].weight.detach().clone()
    torch.optim.AdamW(new.parameters(), lr=1e-3).step()
    assert not torch.equal(before, new.intent_pool.query[1].weight)
    for bad in ({'intent_pooling': 'mean'}, {'intent_target_attention_mix': 1.1}):
        args = dict(kw, intent_target_pooling='current_attention'); args.update(bad)
        try:
            VLAModel(**args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid config accepted')
    print('PASS: default compatibility, causal selector, weighted target/mix endpoints, detached targets, full flow loss backward/update')


if __name__ == '__main__':
    main()
