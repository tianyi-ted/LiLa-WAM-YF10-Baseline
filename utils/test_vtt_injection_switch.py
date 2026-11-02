"""CPU regression test: PYTHONPATH=. python utils/test_vtt_injection_switch.py."""
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
    default = VLAModel(**kw).eval()
    explicit = VLAModel(**kw, inject_task_cond=True).eval()
    indirect = VLAModel(**kw, inject_task_cond=False).eval()
    explicit.load_state_dict(default.state_dict(), strict=True)
    indirect.load_state_dict(default.state_dict(), strict=True)
    state, task, patches = torch.randn(2, 1, 4), torch.randn(2, 32), torch.randn(2, 12, 32)
    noisy, t = torch.randn(2, 8, 4), torch.rand(2)
    sequences = []
    def run(model, vtt):
        handle = model.blocks[0].register_forward_pre_hook(lambda m, args: sequences.append(args[0]))
        try:
            intent = model.predict_global_intent(patches, state, vtt)
            out = model(t=t, noisy_actions=noisy, qpos_history=state,
                        dino_features_list=[patches], task_cond=vtt, global_intent=intent)
            return out, intent
        finally:
            handle.remove()
    a, pred = run(default, task)
    b, _ = run(explicit, task)
    c, indirect_pred = run(indirect, task)
    assert torch.equal(a['final_pred'], b['final_pred'])
    assert torch.equal(sequences[0], sequences[1])
    assert torch.equal(pred, indirect_pred)
    assert sequences[2].shape[1] == sequences[0].shape[1] - 1
    index = 8 + 1 + indirect.num_registers
    expected = indirect.effective_intent_gate() * (
        indirect.intent_to_token(indirect_pred).unsqueeze(1) + indirect.type_emb_intent)
    assert torch.equal(sequences[2][:, index:index+1], expected)
    assert not torch.equal(indirect_pred, indirect.predict_global_intent(patches, state, task + torch.randn_like(task)))
    indirect_pred.square().mean().backward()
    assert indirect.task_cond_proj[1].weight.grad is None
    for param in (indirect.intent_pool.query[1].weight, indirect.intent_predictor.task_proj[1].weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm() > 0
    print('PASS: legacy output/key compatibility, exactly one VTT token removed, VTT still conditions G2 with gradients')


if __name__ == '__main__':
    main()
