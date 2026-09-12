"""L068 complete released Drift-Resilient TabPFN, numeric uncached CPU path.
Built with TabPFN. Independently expanded teaching implementation against
https://github.com/automl/Drift-Resilient_TabPFN at a6e75afb82d13e7abb46deade3669c4106b3d636.
Modified/re-expressed preprocessing, model, data protocol and diagnostics for teaching;
see ../sources/l068-v2/LICENSE.txt and NOTICE.md. No original pretraining generator claim.
"""
import copy,hashlib,importlib.metadata,inspect,json,math,os,time,types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
COMMIT='a6e75afb82d13e7abb46deade3669c4106b3d636'
BASE_URL='https://raw.githubusercontent.com/automl/Drift-Resilient_TabPFN/'+COMMIT+'/'
MODEL_CONFIG=dict(width=192,heads=6,hidden=768,layers=12,classes=10,group_size=2)
CHECKPOINTS={'tabpfn_base_model_2.cpkt': '9ff97d94eae134d708fcd4450e6dd13a2acabcf1b63219e8efc0e087f62fdaa0', 'tabpfn_base_model_1.cpkt': 'fc867cd7da84c7db813f3d68f0aa9f34dec9121d364d6c6c7140063499c1f892', 'tabpfn_base_model_3.cpkt': '0059a69b968e58439ee45b6863c9ce7a03d4019ac120dbfb6c430c767f72ec79', 'tabpfn_dist_ablation_no_t2v_model_1.cpkt': '8882a6d7000e76b7d6bf340492c914204407069240da406e8ebb22b18878375d', 'tabpfn_dist_model_1.cpkt': 'e653d21bd9b2713cf857696f98aabb2a59a4009212d1a63744feccbca94462db', 'tabpfn_dist_model_3.cpkt': 'a9ac82e89b2636999d0d1d7b8da9f7c8dff046ed7921339eec02f804cad369dc', 'tabpfn_dist_model_2.cpkt': '6020bfb897d1ac442aaa8f7ef90985e99740f240b714e79625e33cb205ba0c83'}
PROTOCOL=dict(source_fractions=[.4,.55,.7],id_fraction=.1,temperature=1.,views=1,raw_numeric=True,optimized_preprocessing=False,fp16=False,query_batching=False,pretraining='NOT_RUN',paper_reproduction='INCOMPARABLE')
PRESETS68={'smoke':dict(datasets=['blobs'],seeds=[0],checkpoints=[1],cap_per_domain=12), 'lab':dict(datasets=['electricity','blobs'],seeds=[0],checkpoints=[1],cap_per_domain=None), 'closer':dict(datasets=['electricity','parking','chess','blobs'],seeds=[0,1,2],checkpoints=[1,2,3],cap_per_domain=64), 'full_local':dict(datasets=['electricity','parking','chess','blobs'],seeds=[0,1,2],checkpoints=[1,2,3],cap_per_domain=None)}

def normalize_time(c,n):
    """T×B×1 source min-max; preserve extrapolation, clip only to [-5,6]."""
    if c.ndim!=3 or c.shape[-1]!=1 or not 1<=n<=len(c) or not torch.isfinite(c).all():raise ValueError('Finite time T,B,1 and valid context size')
    lo=c[:n].amin(0,keepdim=True);hi=c[:n].amax(0,keepdim=True)
    span=hi-lo;span=span+(span<1e-16).to(c.dtype)
    return ((c-lo)/span).clamp(-5,6)

def time2vec(c,weight,bias):
    """Learned affine phase: coordinate0 linear, other99 sine."""
    a=nn.functional.linear(c,weight,bias)
    return torch.cat([a[...,:1],a[...,1:].sin()],-1)

def shifted_weights(base,edge_to_relation,selected,shifts):
    """Add edge-specific H(c) outputs only to selected causal relationships."""
    base=np.asarray(base);mapping=np.asarray(edge_to_relation);shifts=np.asarray(shifts)
    if base.ndim!=1 or mapping.shape!=base.shape or shifts.shape[-1]!=len(base):raise ValueError('One relation id and one shift channel per functional edge')
    mask=np.isin(mapping,np.asarray(selected))
    return base+shifts*mask

def temporal_split(domains,source_count,seed,cap_per_domain=None):
    """Fixed domain boundary; label-blind caps; hold out10% of source rows as ID."""
    domains=np.asarray(domains);unique=np.unique(domains)
    if not 2<=source_count<len(unique):raise ValueError('At least two source domains and one future domain')
    rng=np.random.default_rng(seed);train=[];inside=[];future=[]
    for i,d in enumerate(unique):
        ids=rng.permutation(np.flatnonzero(domains==d))
        if cap_per_domain is not None:ids=ids[:cap_per_domain]
        if i<source_count:
            count=max(1,int(np.floor(.1*len(ids))))
            inside.extend(ids[:count]);train.extend(ids[count:])
        else:future.extend(ids)
    return {k:np.asarray(v,dtype=int) for k,v in [('train',train),('id',inside),('ood',future)]}

def dataset_summary(records):
    """Average repetitions within datasets before equally weighting datasets."""
    output=[]
    for name in sorted({r['dataset'] for r in records}):
        for arm in sorted({r['arm'] for r in records}):
            for split in ['id','ood']:
                rows=[r for r in records if r['dataset']==name and r['arm']==arm and r['split']==split]
                if not rows:continue
                item=dict(dataset=name,arm=arm,split=split,n=len(rows))
                for metric in ['accuracy','auc','log_loss']:
                    v=np.array([r[metric] for r in rows if r[metric] is not None],float)
                    item[metric+'_n']=len(v)
                    item[metric+'_mean']=float(v.mean()) if len(v) else None;item[metric+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.
                output.append(item)
    return output

def group_features(x,group_size=2):
    """B×N×F -> B×N×ceil(F/g)×g, preserving row/feature identities."""
    if x.ndim!=3 or group_size<1 or x.shape[-1]<1:raise ValueError('Expected B,N,F and positive group size')
    padding=(-x.shape[-1])%group_size
    return nn.functional.pad(x,(0,padding)).reshape(*x.shape[:2],-1,group_size)

def attention_mix(q,k,v):
    """...×H×receivers×d -> weighted values; K/V may broadcast one head."""
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2]:raise ValueError('Attention shape mismatch')
    # Historical CPU branch rounds the inverse-square-root scalar to float32.
    scale=torch.sqrt(torch.tensor(1.0/q.shape[-1])).to(q.device)
    return ((q@k.transpose(-2,-1))*scale).softmax(-1)@v

class PackedAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.width=width
        self.qkv=nn.Parameter(torch.empty(3,heads,width//heads,width))
        self.out=nn.Parameter(torch.empty(heads,width//heads,width))
        nn.init.normal_(self.qkv,std=.02);nn.init.normal_(self.out,std=.02)
    def forward(self,receivers,senders=None,first_kv=False):
        senders=receivers if senders is None else senders
        q=torch.einsum('...sd,hkd->...hsk',receivers,self.qkv[0])
        weights=self.qkv[1:,:1] if first_kv else self.qkv[1:]
        k,v=torch.einsum('...sd,jhkd->j...hsk',senders,weights)
        mixed=attention_mix(q,k,v)
        return torch.einsum('...hsk,hkd->...sd',mixed,self.out)

def row_attention(h,n,attention):
    """B,N,G,D -> B,N,G,D; context MHA, query first-head-KV MQA."""
    if h.ndim!=4 or not 1<=n<h.shape[1]:raise ValueError('Context/query row split')
    rows=h.transpose(1,2)
    context=rows[:,:,:n];query=rows[:,:,n:]
    context_update=attention(context,context)
    query_update=attention(query,context,first_kv=True)
    return torch.cat([context_update,query_update],2).transpose(1,2)

def postnorm_update(h,update):
    """Every sublayer adds its input BEFORE non-affine LayerNorm(eps=1e-5)."""
    return nn.functional.layer_norm(h+update,(h.shape[-1],),eps=1e-5)

class V2Block(nn.Module):
    def __init__(self,width=192,heads=6,hidden=768):
        super().__init__();self.feature=PackedAttention(width,heads);self.row=PackedAttention(width,heads)
        self.ff1=nn.Linear(width,hidden,bias=False);self.ff2=nn.Linear(hidden,width,bias=False)
    def forward(self,h,n):
        h=postnorm_update(h,self.feature(h))
        h=postnorm_update(h,row_attention(h,n,self.row))
        return postnorm_update(h,self.ff2(nn.functional.gelu(self.ff1(h))))

def numeric_groups(x,n):
    """Exact release order on T×(B G)×2: all-row constant compaction, impute,
    context masked sample z-score with1e-6, clamp, used-feature sqrt scaling, pad.
    Query-dependent constant detection is inherited source behavior and disclosed.
    """
    keep=(x[1:]==x[0]).sum(0)!=(len(x)-1)
    columns=[]
    for b in range(x.shape[1]):
        z=x[:,b,keep[b]]
        if x.shape[1]>1:z=nn.functional.pad(z,(0,x.shape[-1]-z.shape[-1]))
        columns.append(z)
    x=torch.stack(columns,1)
    valid=torch.isfinite(x[:n]);count=valid.to(x.dtype).sum(0)
    mean=torch.where(valid,x[:n],0).sum(0)/(count+1e-10)
    x=torch.where(torch.isfinite(x),x,mean[None])
    valid=~torch.isnan(x[:n]);count=valid.to(x.dtype).sum(0)
    mean=torch.where(valid,x[:n],0).sum(0)/(count+1e-16)
    std=torch.sqrt(torch.nansum((mean[None]-x[:n]).square(),0)/(count+1e-16-1))+1e-6
    if n==1:std=torch.ones_like(std)
    z=((x-mean)/std).clamp(-100,100)
    active=((z[1:]==z[0]).sum(0)!=(len(z)-1)).sum(-1,keepdim=True).clamp_min(1)
    z=z*torch.sqrt(2/active)
    return nn.functional.pad(z,(0,2-z.shape[-1]))

def target_channels(y,total):
    """B×C labels -> B×T×2; imputed ordinal query value plus -2 missing flag."""
    n=y.shape[1];v=nn.functional.pad(y,(0,total-n),value=float('nan')).T[...,None]
    good=torch.isfinite(v[:n]);mean=torch.where(good,v[:n],0).sum(0)/(good.to(v.dtype).sum(0)+1e-10)
    missing=~torch.isfinite(v);v=torch.where(missing,mean[None],v)
    ranks=torch.stack([(v[:,b,:,None]>torch.unique(v[:n,b])).sum(-1) for b in range(v.shape[1])],1).to(v.dtype)
    return torch.cat([ranks,missing.to(v.dtype)*-2],-1).transpose(0,1)

class DriftPFN(nn.Module):
    def __init__(self,variant='dist'):
        super().__init__();self.variant=variant;self.config=dict(MODEL_CONFIG)
        self.time_layer=nn.Linear(1,100) if variant=='dist' else None
        extra={'dist':100,'base':0,'noT2V':1}[variant]
        self.x_encoder=nn.Linear(2+extra,192);self.y_encoder=nn.Linear(2,192);self.position=nn.Linear(48,192)
        self.blocks=nn.ModuleList([V2Block() for _ in range(12)])
        self.head=nn.Sequential(nn.Linear(192,768),nn.GELU(),nn.Linear(768,10))
    def forward(self,x,y,c=None,seed=0,return_trace=False):
        if x.ndim!=3 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not 1<=y.shape[1]<x.shape[1]:raise ValueError('x B,T,F; only context labels y B,C')
        b,t,f=x.shape;n=y.shape[1];g=group_features(x);ng=g.shape[2]
        z=numeric_groups(g.permute(1,0,2,3).reshape(t,b*ng,2),n)
        trace={'numeric':z.detach().clone()}
        if self.variant!='base':
            if c is None or c.shape!=(b,t):raise ValueError('Time indices B,T required')
            ct=normalize_time(c.T[...,None],n)
            tv=time2vec(ct,self.time_layer.weight,self.time_layer.bias) if self.variant=='dist' else ct
            tv=tv[:, :,None].expand(t,b,ng,tv.shape[-1]).reshape(t,b*ng,-1)
            z=torch.cat([z,tv],-1);trace.update(normalized_time=ct.detach().clone(),time_encoding=tv.detach().clone())
        features=self.x_encoder(z).reshape(t,b,ng,192).permute(1,0,2,3)
        generator=torch.Generator(device=x.device).manual_seed(seed)
        ids=torch.randn((ng,48),generator=generator,device=x.device,dtype=x.dtype)
        features=features+self.position(ids)[None,None]
        target=self.y_encoder(target_channels(y,t));h=torch.cat([features,target[:,:,None]],2)
        trace['input']=h.detach().clone()
        for i,block in enumerate(self.blocks):
            h=block(h,n)
            if return_trace:trace['block'+str(i)]=h.detach().clone()
        logits=self.head(h[:,n:,-1]);trace['logits']=logits.detach().clone()
        return (logits,trace) if return_trace else logits

def ensure_file(root,relative,sha=None):
    import urllib.request
    path=Path(root)/'data/cache/l068-release'/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix('.partial');urllib.request.urlretrieve(BASE_URL+relative,part)
        if sha and hashlib.sha256(part.read_bytes()).hexdigest()!=sha:raise ValueError('Download digest mismatch')
        part.replace(path)
    if sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Cached digest mismatch')
    return path

def load_pretrained(root,variant='dist',checkpoint=1):
    family='dist_ablation_no_t2v' if variant=='noT2V' else variant
    name=f'tabpfn_{family}_model_{checkpoint}.cpkt';path=ensure_file(root,'tabpfn/model_cache/'+name,CHECKPOINTS[name])
    saved=torch.load(path,map_location='cpu',weights_only=False);model=DriftPFN(variant)
    xprefix={'dist':'encoder.6.layer.','base':'encoder.4.layer.','noT2V':'encoder.5.layer.'}[variant];mapped={}
    for key,value in saved['state_dict'].items():
        key=key.replace('encoder.5.linear_time_transform.','time_layer.').replace(xprefix,'x_encoder.').replace('y_encoder.2.layer.','y_encoder.').replace('feature_positional_embedding_embeddings.','position.').replace('decoder_dict.standard.','head.')
        key=key.replace('transformer_encoder.layers.','blocks.').replace('self_attn_between_features._w_','feature.').replace('self_attn_between_items._w_','row.').replace('mlp.linear1.','ff1.').replace('mlp.linear2.','ff2.')
        if key in mapped:raise ValueError('Tensor mapping collision')
        mapped[key]=value
    model.load_state_dict(mapped,strict=True);model.eval()
    assert sum(v.numel() for v in mapped.values())==sum(p.numel() for p in model.parameters())
    return model,dict(checkpoint=name,sha256=CHECKPOINTS[name],tensors=len(mapped),parameters=sum(p.numel() for p in model.parameters()),config=saved['config'])

TASK_TYPE_MULTICLASS='multiclass'
DATA_SHA={'elec2.csv': 'e5406c893fe6c727729e1bdb09238424808b0e37e172b3a90aa18ed11dd79d17', 'parking_birmingham.csv': 'f1e28b9c697769a1a05cd20148241b60b1a73fee4907611163efc6fd9d12770a', 'chess.csv': 'c5f2fcaa52a351275223432cef0aa3117b9c8b9a6d5a69a03bf5a91cc88346ab'}

def dataframe_to_distribution_shift_ds(name,df,target,domain_name,task_type,dataset_source,shuffled=False):
    """Same release categorical ordinal codes; preserve complete processed row order."""
    df=df.copy();df[target]=df[target].astype('category')
    cats=df.select_dtypes(['object','category','bool']).columns
    for col in cats:df[col]=df[col].astype('category').cat.codes
    y=df[target].to_numpy().astype(int);c=df[domain_name].to_numpy().astype(int)
    x=df.drop(columns=[target,domain_name])
    return dict(x=x.to_numpy().astype(np.float32),y=y,c=c,features=list(x.columns),name=name,source=dataset_source)

def get_parking_birmingham_data(MODULE_DIR):
    """
    @misc{misc_parking_birmingham_482,
      author       = {Stolfi,Daniel},
      title        = {{Parking Birmingham}},
      year         = {2019},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C51K5Z}
    }

    https://archive.ics.uci.edu/dataset/482/parking+birmingham

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(os.path.join(MODULE_DIR, "data/parking_birmingham.csv"), sep=",")

    # Convert 'LastUpdated' to datetime format
    data["LastUpdated"] = pd.to_datetime(
        data["LastUpdated"], format="%Y-%m-%d %H:%M:%S"
    )

    # Create 'Percentage_Occupied' column and discretize into intervals of 25 percent
    data["Percentage_Occupied"] = (data["Occupancy"] / data["Capacity"]) * 100
    # values smaller than 25 get 0, values between 25 and 50 get 1, values between 50 and 75 get 2, values larger than 75 get 3
    data["Percentage_Occupied"] = np.digitize(
        data["Percentage_Occupied"], bins=[25, 50, 75]
    ).astype(int)

    # Create 'Day', 'Week', and 'Domain' columns
    data["Hour"] = data["LastUpdated"].dt.hour
    data["Day"] = data["LastUpdated"].dt.day
    data["Month"] = data["LastUpdated"].dt.month
    data["Week_Dom"] = (
        data["LastUpdated"].dt.isocalendar().week
        - min(data["LastUpdated"].dt.isocalendar().week)
    ).astype(int)

    # Filter the data to only include the car park with the largest capacity
    data = data[data["SystemCodeNumber"] == "Others-CCCPS133"]

    # Remove 'LastUpdated', 'SystemCodeNumber', 'Week' and 'Year' columns
    data.drop(["LastUpdated", "SystemCodeNumber", "Occupancy"], axis=1, inplace=True)

    return dataframe_to_distribution_shift_ds(
        "Parking Birmingham",
        data,
        "Percentage_Occupied",
        "Week_Dom",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_electricity_data(MODULE_DIR):
    """..
    @Book{ harries1999splice,
        author = { Harries, Michael},
        title = { Splice-2 comparative evaluation: Electricity Pricing},
        publisher = { University of New South Wales, School of Computer Science and Engineering [Sydney] },
        year = { 1999 },
        type = { Book, Online },
        url = { http://nla.gov.au/nla.arc-32869 },
        language = { English },
        subjects = { Machine learning },
        life-dates = { 1999 -  },
        catalogue-url = { https://nla.gov.au/nla.cat-vn3513275 },
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/elec2.csv"),
        sep=",",
        na_values="?",
        skipinitialspace=True,
    )

    # Convert the 'date' column to string type
    df["date"] = df["date"].astype(str)

    # Extract the year, month, and day from the 'date' column
    df["date"] = (
        "19"
        + df["date"].str[:2]
        + "-"
        + df["date"].str[2:4]
        + "-"
        + df["date"].str[4:6]
    )

    # Date ranged from 7 May 1996 to 5 December 1998
    # Create a new 'date' column using the extracted year, month, and day
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Drop nan rows
    df.dropna(inplace=True)

    # Drop half of the rows to reduce the size of the dataset
    df = df[df["half_hour_interval"] % 4 == 0]

    # Create the domain column in which every two week period is considered a new domain
    # Create a grouper object based on 'date' and group by it
    grouper = pd.Grouper(key="date", freq="1W")

    # Create a new column 'domain', each unique group will have a unique identifier
    df["domain"] = df.groupby(grouper).ngroup()

    # Drop the first and last group, since those can be incomplete weeks
    df = df[df["domain"] != 0]
    df = df[df["domain"] != df["domain"].max()]

    # Change the data type of some columns
    df = df.astype(
        {
            "day_of_week": "category",
            "half_hour_interval": "int",
            "nsw_demand": "float",
            "v_demand": "float",
            "transfer": "float",
        }
    )

    # Drop the columns not intended for the model
    df.drop(["date", "nsw_prize", "v_prize"], axis=1, inplace=True)

    # Subsample a range of 15 domains to reduce the size of the dataset even further
    np.random.seed(0)  # Fixing the seed for reproducibility

    range = 15
    max_start = df["domain"].max() - range
    start = np.random.randint(1, max_start + 1)
    end = start + range

    df = df[(df["domain"] >= start) & (df["domain"] < end)]

    return dataframe_to_distribution_shift_ds(
        name="Electricity",
        df=df,
        target="target",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_chess_data(MODULE_DIR):
    """
    @article{vzliobaite2011combining,
      title={Combining similarity in time and space for training set formation under concept drift},
      author={{\v{Z}}liobait{\.e}, Indr{\.e}},
      journal={Intelligent Data Analysis},
      volume={15},
      number={4},
      pages={589--611},
      year={2011},
      publisher={IOS Press}
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/chess.csv"), sep=",", skipinitialspace=True
    )

    # Date ranged from 7 December 2007 to 26 March 2010
    # Build the date column out of the year, month and day column
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Sort by the date
    df = df.sort_values(by="date").reset_index(drop=True)

    # Group every 20 consecutive games as a single domain, this should better track the progress a player makes
    # as the time contains gaps, that are probably not relevant to a player's progress
    df["domain"] = df.index // 20

    # Change the data type of some columns
    cat_columns = ["white/black", "type", "outcome"]
    df[cat_columns] = df[cat_columns].astype("category")

    # Drop the columns not intended for the model
    df.drop(["date"], axis=1, inplace=True)

    return dataframe_to_distribution_shift_ds(
        name="Chess",
        df=df,
        target="outcome",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_intersecting_blobs(
    num_domains=14,
    num_samples=40,
    random_state=0,
    name="Intersecting Blobs Dataset",
):
    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Initialize the centers and standard deviations of the blobs
    centers = np.array([[-16, 0.0], [6.0, 12.0], [4.0, -8.0]])
    std_devs = np.array([1.1, 0.9, 1.0])

    xs = []
    ys = []
    domains = []

    # For each domain
    for i in range(num_domains):
        # Move the blobs
        centers += (
            np.random.normal(
                loc=[2.0, 0.2, -1, -2.5, -0.1, 0.5], scale=0.1, size=6
            ).reshape(3, 2)
            * 0.9
        )

        # Generate data points for each blob
        for idx, (center, std_dev) in enumerate(zip(centers, std_devs)):
            points = np.random.normal(loc=center, scale=std_dev, size=(num_samples, 2))
            xs.append(points)
            ys.extend([idx] * num_samples)
            domains.extend([i] * num_samples)

        # Vary the standard deviation of the blobs
        std_devs += np.random.normal(loc=0.0, scale=0.1, size=3)
        std_devs = np.clip(
            std_devs, 0.1, np.inf
        )  # Ensure that the standard deviations remain positive

    # Concatenate the datasets
    x = np.vstack(xs)
    y = np.array(ys)
    domain = np.array(domains)

    features = [f"Feature{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=features)
    df["Label"] = y
    df["Domain"] = domain

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )

def load_dataset(root,name):
    if name=='blobs':return get_intersecting_blobs()
    filenames={'electricity':'elec2.csv','parking':'parking_birmingham.csv','chess':'chess.csv'}
    file=filenames[name];path=ensure_file(root,'tabpfn/datasets/data/'+file,DATA_SHA[file])
    loader={'electricity':get_electricity_data,'parking':get_parking_birmingham_data,'chess':get_chess_data}[name]
    return loader(str(path.parent.parent))

def sample_scm(seed=0,domains=8,rows=96):
    """Paper-grounded reconstruction: scalar functional graph for four causal nodes,
    nonlinear two-layer mechanisms and shared nonlinear second-order network.
    The exact original pretraining sampler/hyperprior is unavailable in the release.
    """
    rng=np.random.default_rng(seed)
    # Causal nodes: U, V, X, Y. Functional expansion uses two hidden units per assignment.
    # Edges: U->Xhidden0/1, V->Xhidden0/1; Xhidden->X; U->Yhidden0/1,
    # X->Yhidden0/1; Yhidden->Y. Hidden-output edges belong to -1, unchanged.
    src=np.array([0,0,1,1,2,3,0,0,4,4,5,6]);dst=np.array([2,3,2,3,4,4,5,6,5,6,7,7])
    relation=np.array([0,0,1,1,-1,-1,2,2,3,3,-1,-1])
    base=rng.normal(0,1,len(src));selected=np.array([0,3])
    # H(c) has shared hidden causes, then correlated output channels for every functional edge.
    a=rng.normal(size=(1,5));b=rng.normal(size=5);w=rng.normal(size=(5,4));v=rng.normal(size=(4,len(src)))
    c=np.linspace(0,1.75,domains);shared=np.tanh(c[:,None]@a+b);hidden=np.sin(shared@w)
    shifts=.6*(hidden@v);weights=shifted_weights(base,relation,selected,shifts)
    noise=rng.normal(size=(domains,rows,8));values=np.zeros_like(noise)
    values[:,:,:2]=noise[:,:,:2]
    for node in range(2,8):
        incoming=np.flatnonzero(dst==node)
        z=sum(weights[:,e,None]*values[:,:,src[e]] for e in incoming)
        values[:,:,node]=np.tanh(z+.15*noise[:,:,node]) if node in [2,3,5,6] else z+.15*noise[:,:,node]
    x=values[:,:,[0,1,4]].reshape(-1,3).astype(np.float32);score=values[:,:,7]
    y=(score>0).astype(int).ravel()
    return dict(x=x,y=y,c=np.repeat(np.arange(domains),rows),features=['U','V','X'],name='Reconstructed SCM',source='synthetic',trace=dict(seed=seed,base=base.tolist(),edge_source=src.tolist(),edge_target=dst.tolist(),edge_to_relation=relation.tolist(),selected=selected.tolist(),domains=c.tolist(),second_order_a=a.tolist(),second_order_b=b.tolist(),second_order_w=w.tolist(),second_order_v=v.tolist(),shared=shared.tolist(),hidden=hidden.tolist(),shifts=shifts.tolist(),weights=weights.tolist(),sample_nodes=values[:,0].tolist(),class_fraction=y.reshape(domains,rows).mean(1).tolist(),scope='Reconstruction of Algorithm1 mechanisms; not original prior distribution or pretraining'))

def model_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str((tuple(tensor.shape),tensor.dtype)).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def stable_value(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if v is Ellipsis:return 'Ellipsis'
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(list,tuple)):return [stable_value(i) for i in v]
    if isinstance(v,dict):return {str(k):stable_value(i) for k,i in v.items()}
    if isinstance(v,(set,frozenset)):return sorted([stable_value(i) for i in v],key=repr)
    if isinstance(v,bytes):return v.hex()
    if isinstance(v,Path):return str(v)
    if isinstance(v,types.CodeType):return {k:stable_value(getattr(v,k)) for k in ['co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags']}
    if inspect.isfunction(v):return dict(code=stable_value(v.__code__),defaults=stable_value(v.__defaults__),kwdefaults=stable_value(v.__kwdefaults__),closure=stable_value([c.cell_contents for c in v.__closure__ or []]))
    if inspect.isclass(v):return v.__module__+'.'+v.__qualname__
    raise TypeError('Unbound identity value '+str(type(v)))

def model_runtime_identity(model):
    """Bind actual module types, forward methods and non-weight runtime settings.

    Instance overrides and hooks are unsupported by this certified inference path;
    reject them rather than pretending an arbitrary callback has been audited.
    """
    records={}
    for name,module in model.named_modules():
        if 'forward' in vars(module):raise RuntimeError('Instance forward override invalidates source certification')
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:raise RuntimeError('Module hooks invalidate source certification')
        for parameter in module.parameters(recurse=False):
            if parameter._backward_hooks or getattr(parameter,'_post_accumulate_grad_hooks',None):raise RuntimeError('Parameter gradient hooks invalidate source certification')
        method=module.forward.__func__
        settings={k:stable_value(v) for k,v in vars(module).items() if not k.startswith('_')}
        records[name]=dict(type=type(module).__module__+'.'+type(module).__qualname__,forward=stable_value(method),settings=settings)
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()

def kernel_identity(namespace,root=None):
    """Recursively bind live code, nested generator globals, defaults and constants."""
    roots=['normalize_time','time2vec','shifted_weights','temporal_split','dataset_summary','DriftPFN','load_pretrained','load_dataset','sample_scm','predict','run_experiment','model_runtime_identity','model_digest']
    seen={};allowed={namespace['DriftPFN'].__module__,namespace['run_experiment'].__module__}
    def visit(name,obj):
        if name in seen:return
        if inspect.isclass(obj):
            seen[name]={'class':obj.__qualname__}
            for k,v in vars(obj).items():
                if isinstance(v,(staticmethod,classmethod)):v=v.__func__
                if inspect.isfunction(v):visit(name+'.'+k,v)
        elif inspect.isfunction(obj):
            seen[name]=stable_value(obj)
            def dependency(v,n):
                if (inspect.isfunction(v) or inspect.isclass(v)) and v.__module__ in allowed:visit(n,v)
            def scan(code):
                for n in code.co_names:dependency(obj.__globals__.get(n),n)
                for c in code.co_consts:
                    if isinstance(c,types.CodeType):scan(c)
            scan(obj.__code__)
            for i,v in enumerate(obj.__defaults__ or []):dependency(v,name+'.default'+str(i))
            for k,v in (obj.__kwdefaults__ or {}).items():dependency(v,name+'.kwdefault.'+k)
    for name in roots:visit(name,namespace[name])
    result=dict(operators={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in seen.items()},constants={k:stable_value(namespace[k]) for k in ['MODEL_CONFIG','PROTOCOL','PRESETS68','CHECKPOINTS','DATA_SHA','COMMIT','BASE_URL','TASK_TYPE_MULTICLASS']},versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','scikit-learn','pandas']})
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest();return result

def predict(model,x,y,c,context,query,arm,seed=17):
    """All heldout queries in one call; time as raw feature for baseline-time control."""
    ids=np.concatenate([context,query]);raw=np.asarray(x[ids],np.float32);times=np.asarray(c[ids],np.float32)
    if arm=='base_time':raw=np.column_stack([raw,times])
    if arm=='drift_zero':times=np.zeros_like(times)
    xx=torch.from_numpy(raw)[None];yy=torch.tensor(y[context],dtype=torch.float32)[None];cc=torch.from_numpy(times)[None]
    classes=int(np.max(y[context]))+1
    if not np.array_equal(np.unique(y[context]),np.arange(classes)):raise ValueError('Training class axis must be contiguous')
    with torch.no_grad():z=model(xx,yy,cc,seed=seed)[0,:,:classes];p=(z/PROTOCOL['temperature']).softmax(-1)
    return p.numpy()

def metrics(y,p):
    classes=p.shape[1]
    auc=roc_auc_score(y,p[:,1]) if classes==2 and len(np.unique(y))==2 else roc_auc_score(y,p,multi_class='ovr',labels=np.arange(classes)) if len(np.unique(y))==classes else None
    return dict(accuracy=float(accuracy_score(y,p.argmax(1))),auc=None if auc is None else float(auc),log_loss=float(log_loss(y,p,labels=np.arange(classes))))

def run_experiment(root,config=None,namespace=None):
    """Fresh actual checkpoint evidence, paired row IDs and strict source/load identity.
    Three paired repetitions vary split sampling AND pretrained checkpoint id together;
    they are not independent datasets or a factorial estimate of either variance.
    """
    cfg=copy.deepcopy(PRESETS68['lab'] if config is None else config);ns=globals() if namespace is None else namespace
    identity=kernel_identity(ns,root);records=[];datasets={};models={};initial={};started=time.perf_counter()
    checkpoint_ids=cfg['checkpoints']
    for j,seed in enumerate(cfg['seeds']):
      ck=checkpoint_ids[j%len(checkpoint_ids)]
      for v in ['base','dist','noT2V']:
        key=(v,1 if v=='noT2V' else ck)
        if key not in models:
          model,meta=load_pretrained(root,*key);models[key]=model
          initial[str(key)]=dict(checkpoint=meta['checkpoint'],sha256=meta['sha256'],parameters=meta['parameters'],tensors=meta['tensors'],weights=model_digest(model),runtime=model_runtime_identity(model))
      for name in cfg['datasets']:
        data=load_dataset(root,name);x,y,c=data['x'],data['y'],data['c']
        count=max(2,int(len(np.unique(c))*PROTOCOL['source_fractions'][j%3]));split=temporal_split(c,count,seed,cfg['cap_per_domain'])
        for kind,ids in split.items():
          if not set(np.unique(y[ids])).issubset(set(np.unique(y[split['train']]))):raise ValueError('New class in '+name+' '+kind)
        datasets[name]=dict(shape=list(x.shape),features=data['features'],source=data['source'],x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),c_sha256=hashlib.sha256(c.tobytes()).hexdigest(),domains=np.unique(c).tolist(),source_count=count)
        for arm in ['base_no_time','base_time','drift','drift_zero','noT2V']:
          v='base' if arm.startswith('base') else 'noT2V' if arm=='noT2V' else 'dist';model=models[(v,1 if v=='noT2V' else ck)]
          for kind in ['id','ood']:
            tick=time.perf_counter();ids=split[kind];p=predict(model,x,y,c,split['train'],ids,arm,17+seed)
            rows=dict(dataset=name,seed=seed,checkpoint=1 if v=='noT2V' else ck,arm=arm,split=kind,source_count=count,context_ids=split['train'].tolist(),query_ids=ids.tolist(),query_domains=c[ids].tolist(),targets=y[ids].tolist(),probabilities=p.tolist(),feature_seed=17+seed,seconds=time.perf_counter()-tick,**metrics(y[ids],p))
            # Per-domain probability errors expose a changing horizon; no domain-level pseudo replication.
            rows['domain_metrics']=[dict(domain=int(d),rows=int(sum(c[ids]==d)),**metrics(y[ids][c[ids]==d],p[c[ids]==d])) for d in np.unique(c[ids])]
            records.append(rows)
          print(name,seed,ck,arm,records[-1]['accuracy'],records[-1]['log_loss'],flush=True)
    diagnostics=[]
    for seed in cfg['seeds']:
      data=sample_scm(seed);spl=temporal_split(data['c'],5,seed,32)
      for arm in ['base_time','drift']:
        model=models[('base' if arm=='base_time' else 'dist',checkpoint_ids[0])]
        p=predict(model,data['x'],data['y'],data['c'],spl['train'],spl['ood'],arm,17+seed)
        diagnostics.append(dict(seed=seed,arm=arm,trace=data['trace'],context_ids=spl['train'].tolist(),query_ids=spl['ood'].tolist(),targets=data['y'][spl['ood']].tolist(),probabilities=p.tolist(),**metrics(data['y'][spl['ood']],p)))
    for key,model in models.items():
      assert initial[str(key)]['weights']==model_digest(model) and initial[str(key)]['runtime']==model_runtime_identity(model),'Inference changed model state'
    assert identity==kernel_identity(ns,root),'Live namespace changed during run'
    return dict(status='PASS',config=cfg,protocol=PROTOCOL,kernel_identity=identity,models=initial,datasets=datasets,records=records,summary=dataset_summary(records),scm_diagnostics=diagnostics,seconds=time.perf_counter()-started,paper_reproduction='INCOMPARABLE')
