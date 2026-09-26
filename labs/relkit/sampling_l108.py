"""Time-windowed sampling from a timestamp-sorted compressed adjacency index.
Course extension to Xu et al. 2020 §3.4; not the historical released sampler.
"""
# %% Imports
import time
import numpy as np

# %% TODO 1 — define the legal interval

def window_bounds(sorted_times, cutoff, window=np.inf):
    """Return [start,end) positions for timestamps in [cutoff-window,cutoff)."""
    if not np.isfinite(cutoff) or not window>0:
        raise ValueError('Finite cutoff and positive window required')
    return (int(np.searchsorted(sorted_times,cutoff-window,side='left')),
            int(np.searchsorted(sorted_times,cutoff,side='left')))

# %% TODO 2 — spend a bounded neighbor budget

def choose_positions(count, fanout, policy, uniforms):
    """Relative positions, with -1 for padding; uniform draws use replacement."""
    if fanout<1 or count<0 or policy not in ('uniform','recent'):
        raise ValueError('Invalid sampling configuration')
    if count==0:return np.full(fanout,-1,dtype=np.int64)
    if policy=='uniform':
        u=np.asarray(uniforms)
        if u.shape!=(fanout,) or not ((u>=0)&(u<1)).all():raise ValueError('Invalid variates')
        return np.floor(u*count).astype(np.int64)
    positions=np.arange(count-fanout,count)
    return np.where(positions>=0,positions,-1)

# %% TODO 3 — predict recursive expansion

def expansion_size(layers, fanout):
    """Tree upper bound per root: 1 + k + ... + k**L, before deduplication."""
    if layers<0 or fanout<1:raise ValueError('Invalid depth or fanout')
    return sum(fanout**depth for depth in range(layers+1))

# %% Compressed temporal rows — one flat record array, offsets delimit nodes

class TemporalIndex:
    def __init__(self, events, n_nodes):
        u=np.asarray(events['u'],dtype=np.int64);v=np.asarray(events['v'],dtype=np.int64)
        t=np.asarray(events['t'],dtype=np.float64);e=np.asarray(events['e'],dtype=np.int64)
        if not (u.shape==v.shape==t.shape==e.shape) or not np.isfinite(t).all():raise ValueError('Invalid events')
        if len(u) and (min(u.min(),v.min())<1 or max(u.max(),v.max())>=n_nodes):raise ValueError('Node 0 is padding')
        owner=np.r_[u,v];neighbor=np.r_[v,u];ts=np.r_[t,t];edges=np.r_[e,e]
        order=np.lexsort((edges,ts,owner))
        # Sentinel lets empty output gathers remain valid even for an empty graph.
        self.nodes=np.r_[0,neighbor[order]].astype(np.int64)
        self.edges=np.r_[0,edges[order]].astype(np.int64)
        self.times=np.r_[0.,ts[order]]
        self.offsets=np.r_[1,1+np.cumsum(np.bincount(owner,minlength=n_nodes))]
        self.n_nodes=n_nodes
        self.policy='uniform';self.window=np.inf
        self.audit={'queries':0,'sampled_records':0,'nonpast_records':0,'expired_records':0}

    @property
    def array_bytes(self):
        return sum(a.nbytes for a in (self.nodes,self.edges,self.times,self.offsets))

    def lower_bound(self, nodes, boundaries):
        """Batched binary search: each query searches only its node's row."""
        lo=self.offsets[nodes].copy();hi=self.offsets[nodes+1].copy()
        while np.any(lo<hi):
            active=lo<hi;mid=(lo+hi)//2
            probe=self.times[np.minimum(mid,len(self.times)-1)]
            right=active & (probe<boundaries)
            lo=np.where(right,mid+1,lo);hi=np.where(active & ~right,mid,hi)
        return lo

    def sample(self,nodes,cutoffs,fanout=20,policy='uniform',window=np.inf,uniforms=None):
        """Bounded [B,k] outputs (node,event,time); masks encoded by event 0."""
        nodes=np.asarray(nodes,dtype=np.int64);cutoffs=np.asarray(cutoffs,dtype=float)
        if nodes.shape!=cutoffs.shape or nodes.ndim!=1 or not np.isfinite(cutoffs).all():raise ValueError('Invalid queries')
        if np.any(nodes<0) or np.any(nodes>=self.n_nodes):raise ValueError('Unknown node')
        if fanout<1 or policy not in ('uniform','recent') or not window>0:raise ValueError('Invalid config')
        start=self.lower_bound(nodes,cutoffs-window);end=self.lower_bound(nodes,cutoffs)
        count=end-start
        draws=np.random.random((len(nodes),fanout)) if uniforms is None else np.asarray(uniforms)
        if draws.shape!=(len(nodes),fanout) or not ((draws>=0)&(draws<1)).all():raise ValueError('Invalid variates')
        if policy=='uniform':
            pos=start[:,None]+np.floor(draws*count[:,None]).astype(np.int64)
            valid=np.broadcast_to(count[:,None]>0,pos.shape)
        else:
            pos=end[:,None]+np.arange(-fanout,0)
            valid=pos>=start[:,None]
        pos=np.where(valid,pos,0)
        # Stable chronological output; padding precedes real history, including time zero.
        ordering=np.argsort(np.where(valid,self.times[pos],-np.inf),axis=1,kind='stable')
        pos=np.take_along_axis(pos,ordering,axis=1)
        result=(self.nodes[pos].astype(np.int32),self.edges[pos].astype(np.int32),self.times[pos].astype(np.float32))
        mask=result[1]!=0
        self.audit['queries']+=len(nodes);self.audit['sampled_records']+=int(mask.sum())
        self.audit['nonpast_records']+=int(np.sum(mask & (self.times[pos]>=cutoffs[:,None])))
        self.audit['expired_records']+=int(np.sum(mask & (self.times[pos]<cutoffs[:,None]-window)))
        return result

    def sample_scalar(self,nodes,cutoffs,fanout=20,policy='uniform',window=np.inf,uniforms=None):
        """Readable reference: student's boundary and policy are on this live path."""
        draws=np.random.random((len(nodes),fanout)) if uniforms is None else uniforms
        result=[np.zeros((len(nodes),fanout),dtype=d) for d in (np.int32,np.int32,np.float32)]
        for i,(node,cutoff) in enumerate(zip(nodes,cutoffs)):
            base,stop=self.offsets[node:node+2]
            lo,hi=window_bounds(self.times[base:stop],cutoff,window)
            chosen=choose_positions(hi-lo,fanout,policy,draws[i]);valid=chosen>=0
            positions=np.where(valid,base+lo+chosen,0)
            order=np.argsort(np.where(valid,self.times[positions],-np.inf),kind='stable');positions=positions[order]
            for out,arr in zip(result,(self.nodes,self.edges,self.times)):out[i]=arr[positions]
        return tuple(result)

    def get_temporal_neighbor(self,nodes,cutoffs,num_neighbors=20):
        """TGAT adapter: recursive model calls supply each edge's own timestamp."""
        return self.sample(nodes,cutoffs,num_neighbors,self.policy,self.window)

# %% Full data sampler benchmark — same questions and variates, no model inference

def benchmark_sampler(events,n_nodes,batch_size=200,repeats=3):
    start=time.perf_counter();index=TemporalIndex(events,n_nodes);construction=time.perf_counter()-start
    queries=events['u'];cutoffs=events['t'];draws=np.random.default_rng(108).random((len(queries),20))
    for method in (index.sample_scalar,index.sample):method(queries[:batch_size],cutoffs[:batch_size],uniforms=draws[:batch_size])
    timings={'scalar':[],'batched':[]};equal=True
    for i in range(0,len(queries),batch_size):
        q=queries[i:i+batch_size];t=cutoffs[i:i+batch_size];u=draws[i:i+batch_size]
        a=index.sample_scalar(q,t,uniforms=u);b=index.sample(q,t,uniforms=u)
        equal &= all(np.array_equal(x,y) for x,y in zip(a,b))
    assert equal,'An optimization changed sampled events'
    # Alternate order to reduce systematic warm-cache/order confounding.
    for repeat in range(repeats):
      for name in (['scalar','batched'] if repeat%2==0 else ['batched','scalar']):
        method=index.sample_scalar if name=='scalar' else index.sample
        start=time.perf_counter()
        for i in range(0,len(queries),batch_size):method(queries[i:i+batch_size],cutoffs[i:i+batch_size],uniforms=draws[i:i+batch_size])
        timings[name].append(time.perf_counter()-start)
    return {'queries':len(queries),'batch_size':batch_size,'repeats':repeats,'construction_seconds':construction,'index_array_bytes':index.array_bytes,'timing_seconds':timings,'samples_exactly_equal':bool(equal),'scope':'CPU sampling only; fixed precomputed random variates; array bytes exclude allocator and temporary arrays'}
