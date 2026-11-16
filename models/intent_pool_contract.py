"""Narrow teacher compatibility bridge for the opt-in late-pool student.

Student checkpoints/resume retain their FULL architecture contract. Only the
comparison against the existing single-query teacher is canonicalized here.
No image/state/VTT/stats/horizon/DiT setting may be discarded.
"""
from copy import deepcopy


LATE_POOL = 'task_attention_late'
LATE_POOL_FIELDS = ('pool_num_queries', 'pool_interaction_depth')


def uses_late_pool(contract):
    return contract.get('model', {}).get('global_intent', {}).get('pooling') == LATE_POOL


def teacher_input_contract(contract):
    if not uses_late_pool(contract):
        return contract
    teacher_contract = deepcopy(contract)
    intent = teacher_contract['model']['global_intent']
    intent['pooling'] = 'task_attention'
    for key in LATE_POOL_FIELDS:
        intent.pop(key, None)
    return teacher_contract
