"""Run with PYTHONPATH=. python utils/test_g2_nogate.py (CPU only)."""
import torch
from models.vla_model_fm import VLAModel


def main():
    torch.manual_seed(0)
    kw = dict(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1,
              action_len=8, num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4,
              adapter_depth=1, task_cond_dim=32, use_global_intent=True,
              inject_global_intent=True, intent_hidden_dim=32, intent_gate_init=0.1,
              intent_pooling='task_attention', intent_pool_hidden_dim=32,
              intent_pool_grid_size=(3, 4))
    old = VLAModel(**kw).eval()
    explicit = VLAModel(**kw, intent_use_gate=True).eval()
    direct = VLAModel(**kw, intent_use_gate=False).eval()
    explicit.load_state_dict(old.state_dict(), strict=True)
    direct.load_state_dict(old.state_dict(), strict=True)
    assert old.intent_gate.requires_grad and not direct.intent_gate.requires_grad
    state, task, patches = torch.randn(2, 1, 4), torch.randn(2, 32), torch.randn(2, 12, 32)
    noisy, t = torch.randn(2, 8, 4), torch.rand(2)
    captures = []
    def run(model):
        handle = model.blocks[0].register_forward_pre_hook(lambda m, args: captures.append(args[0]))
        try:
            pred = model.predict_global_intent(patches, state, task)
            return model(t=t, noisy_actions=noisy, qpos_history=state,
                         dino_features_list=[patches], task_cond=task, global_intent=pred), pred
        finally:
            handle.remove()
    a, _ = run(old)
    b, _ = run(explicit)
    c, pred = run(direct)
    assert torch.equal(a['final_pred'], b['final_pred'])
    assert torch.equal(captures[0], captures[1])
    index = 8 + 1 + old.num_registers + 1  # actions, state, registers, VTT
    expected = direct.intent_to_token(pred).unsqueeze(1) + direct.type_emb_intent
    assert torch.equal(captures[2][:, index:index+1], expected)
    assert torch.equal(captures[0][:, index:index+1], old.effective_intent_gate() * expected)
    assert direct.effective_intent_gate().item() == 1.0
    captures[2][:, index:index+1].square().mean().backward()
    assert direct.intent_gate.grad is None
    grad = direct.intent_pool.query[1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    before = direct.intent_pool.query[1].weight.detach().clone()
    torch.optim.AdamW(direct.parameters(), lr=1e-3).step()
    assert not torch.equal(before, direct.intent_pool.query[1].weight)
    print('PASS: strict key compatibility, unchanged default gate path, direct token injection, gradient/optimizer step')


if __name__ == '__main__':
    main()
