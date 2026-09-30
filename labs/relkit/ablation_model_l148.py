"""Explicit interventions on the pinned L117 RDL model; course extensions."""
import torch
import numpy as np
from relkit.rdl_l117 import Model
from relkit.ablation_l148 import history_mask,ARMS

def restrict_history(batch,entity,window=365*86400):
    """Prune AFTER temporal sampling; no replacement sampling. Keep seed order."""
    cutoffs=batch[entity].seed_time
    masks={};maps={};removed=0
    for kind in batch.node_types:
        n=batch[kind].num_nodes
        roots=torch.zeros(n,dtype=torch.bool,device=cutoffs.device)
        if kind==entity:roots[:len(cutoffs)]=True
        dated='time' in batch[kind]
        times=batch[kind].time if dated else torch.zeros(n,device=cutoffs.device,dtype=torch.long)
        keep=history_mask(cutoffs.cpu().numpy(),times.cpu().numpy(),batch[kind].batch.cpu().numpy(),np.full(n,not dated),roots.cpu().numpy(),window)
        mask=torch.as_tensor(keep,device=cutoffs.device)
        masks[kind]=mask;mapping=torch.full((n,),-1,device=cutoffs.device,dtype=torch.long)
        mapping[mask]=torch.arange(int(mask.sum()),device=cutoffs.device);maps[kind]=mapping
        removed+=n-int(mask.sum())
    out=batch.clone()
    for kind,mask in masks.items():
        out[kind].tf=batch[kind].tf[mask]
        for key in ['time','batch','n_id']:
            if key in batch[kind]:out[kind][key]=batch[kind][key][mask]
        out[kind].num_nodes=int(mask.sum())
    for (src,rel,dst),edges in batch.edge_index_dict.items():
        keep=masks[src][edges[0]]&masks[dst][edges[1]]
        out[(src,rel,dst)].edge_index=torch.stack([maps[src][edges[0,keep]],maps[dst][edges[1,keep]]])
    return out,removed

class AblationModel(Model):
    def __init__(self,*args,arm='full',**kwargs):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__(*args,**kwargs);self.arm=arm;self.removed_nodes=0
        if arm in ['encoder','combined']:
            for encoder in self.encoder.encoders.values():
                # Keep initialization of shared weights identical. Delete blocks2–4.
                encoder.backbone=torch.nn.Sequential(encoder.backbone[0])

    def forward(self,batch,entity_table):
        if self.arm=='history':
            batch,removed=restrict_history(batch,entity_table);self.removed_nodes+=removed
        if self.arm in ['messages','combined']:
            batch=batch.clone()
            for edge in batch.edge_types:batch[edge].edge_index=batch[edge].edge_index[:,:0]
        return super().forward(batch,entity_table)
