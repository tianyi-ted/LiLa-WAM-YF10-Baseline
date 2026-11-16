"""Offline future interventions for the privileged teacher, NOT rollout success."""
from contextlib import nullcontext
from collections import defaultdict
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from train_planwam import input_contract, load_frozen_teacher, validate_config, sha256
from dataset_native import NativeMetaWorldDataset
from dataset import collate_fn
from models.model_runner import ModelFactory, VLAWrapper
from models.planwam_g2 import planwam_flow_loss
from models.vla_model_fm import fixed_layer_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow-smoke-checkpoint', action='store_true')
    a = p.parse_args()
    if a.batches < 1 or a.batch_size < 1:
        p.error('batches and batch-size must be positive')
    cfg = OmegaConf.load(a.config)
    if validate_config(cfg) != 'teacher':
        p.error('provide teacher configuration')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    rng = random.Random(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    d,c,m = cfg.dataset,cfg.common,cfg.model
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    vision, hidden, regs, patch = ModelFactory.create_vision_encoder(m.vision_encoder.checkpoint_path,dtype,str(device))
    action = ModelFactory.create_action_model(cfg,hidden,len(m.vision_encoder.feat_layers),task_cond_dim=hidden,patch_size=patch,num_register_tokens=regs).to(device=device,dtype=dtype)
    action.load_state_dict(ck['model_state'],strict=True); action.keep_control_gates_fp32(); action.eval()
    teacher = load_frozen_teacher(a.checkpoint,input_contract(cfg,a.stats),ck['posterior_spec'],device,dtype,a.allow_smoke_checkpoint)
    wrapper = VLAWrapper(vision,action,cfg.training.time_sampler,m.vision_encoder.feat_layers,bool(m.vision_encoder.include_cls_register),regs,device,dtype,a.stats,None,int(m.future_feat.target_layer)).to(device).eval()
    ds = NativeMetaWorldDataset(str(d.dataset_dir),list(d.task_names),int(c.action_chunk_size),int(c.state_dim),tuple(d.image_size),str(d.task_cond_dir),True,int(d.future_frame_offset),False,camera_name=str(d.camera_names[0]))
    episode_anchors, task_episodes = defaultdict(list),defaultdict(list)
    for i,(episode,anchor) in enumerate(ds.indices): episode_anchors[episode].append(i)
    for episode,item in enumerate(ds.episodes): task_episodes[item['task']].append(episode)
    if any(len(v)<2 for v in task_episodes.values()):
        raise ValueError('same-task wrong-future control needs at least two episodes per task')
    metrics = defaultdict(list)
    sample_ids = []
    for _ in range(a.batches):
        indices = [rng.randrange(len(ds)) for _ in range(a.batch_size)]
        donors = []
        for i in indices:
            ep,_ = ds.indices[i]
            candidates = [e for e in task_episodes[ds.episodes[ep]['task']] if e!=ep]
            donors.append(rng.choice(episode_anchors[rng.choice(candidates)]))
        sample_ids.append(dict(anchors=[list(ds.indices[i]) for i in indices],donors=[list(ds.indices[i]) for i in donors]))
        batch = collate_fn([ds[i] for i in indices])
        wrong = collate_fn([ds[i] for i in donors])['future_pixel_values']
        amp = torch.amp.autocast('cuda',dtype=dtype) if device.type=='cuda' else nullcontext()
        with torch.no_grad(), amp:
            features = wrapper.get_vision_features(batch['pixel_values'])
            current = wrapper.get_current_intent_features(features)
            future = wrapper.get_future_target_features(batch['future_pixel_values'])
            swapped = wrapper.get_future_target_features(wrong)
            state = wrapper.normalize_state(batch['state'].to(device,dtype).unsqueeze(1))
            task = batch['task_cond'].to(device,dtype)
            actions = wrapper.normalize_action(batch['action_sequence'].to(device,dtype))
            noise = torch.randn_like(actions)
            true_z = fixed_layer_norm(teacher(current,future,state,task,int(d.future_frame_offset)).float())
            for name,f in [('correct',future),('future_equals_current',current),('same_task_other_episode',swapped)]:
                z = fixed_layer_norm(teacher(current,f,state,task,int(d.future_frame_offset)).float())
                metrics[name+'/latent_mse_vs_correct'].append((z-true_z).square().mean().item())
                for tau in (0.1,0.5,0.9):
                    _,info = planwam_flow_loss(action,current,f,features,state,task,actions,'teacher',
                                              posterior=teacher,horizon=int(d.future_frame_offset),
                                              noise=noise,flow_time=torch.full((a.batch_size,),tau,device=device))
                    metrics[f'{name}/action_mse_tau_{tau}'].append(info['loss_action'])
    result = dict(kind='offline_teacher_intervention_not_rollout_success',checkpoint=str(Path(a.checkpoint).resolve()),
                  checkpoint_sha256=sha256(a.checkpoint),seed=a.seed,batches=a.batches,batch_size=a.batch_size,
                  sample_ids=sample_ids,metrics={k:dict(mean=float(np.mean(v)),batch_values=v) for k,v in metrics.items()})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['metrics'],indent=2)); print('Saved',out)


if __name__=='__main__': main()
