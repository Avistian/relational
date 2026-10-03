import yaml
import numpy as np
import os.path as osp
import torch
import copy
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("srcpath", type=str)
parser.add_argument("dstpath", type=str)
parser.add_argument("--ncpu", type=int, default=16)
args = parser.parse_args()

srcpath = args.srcpath #"griffin_datasets/joint-v52-pk-r2n/"#"tpberta-reg-r2n-combine/"#"./griffin_datasets/tpberta-bin-r2n-combine/"
dstpath = args.dstpath #"newdatasets/newds_v52_pk/"

os.makedirs(dstpath, exist_ok=True)

with open(osp.join(srcpath, "metadata.yaml"), "r") as f:
    config: dict = yaml.safe_load(f)

def buildnodemeta(config):
    nodelist = {}
    featlist = {}
    for _ in config["graph"]["nodes"]:
        nodenum, nodetype = _["num"], _["type"]
        assert nodetype not in nodelist
        nodelist[nodetype] = {"num": nodenum, "feat": []}
    for _ in config["graph"]["feature_data"]:
        assert _["domain"] == "node"
        assert _['name'] == '__timestamp__'
        nodetype, path  = _["type"], _["path"]
        assert nodetype in nodelist
        nodelist[nodetype]["timestamp"] = path
    for nodetype in nodelist:
        assert "timestamp" in nodelist[nodetype]
    for _ in config["feature_data"]:
        assert _["domain"] == "node"
        nodetype, path, extrafield, name = _["type"], _["path"], _["extra_fields"], _["name"]
        assert nodetype in nodelist
        nameemb = np.array(extrafield.pop("name_emb"))
        # nameemb = None
        is_target_column = extrafield.get("is_target_column", False)
        dtype = extrafield.get("dtype")
        num_categories = extrafield.get("num_categories", 0)
        name = f"{nodetype}___{name}"
        assert name not in featlist
        featlist[name] = (path, (is_target_column, dtype, num_categories), nameemb)
        nodelist[nodetype]['feat'].append(name)
    for nodetype in nodelist:
        featnamelist = copy.deepcopy(nodelist[nodetype]['feat'])
        for feat in featnamelist:
            if "Griffin_text_" in feat:
                origin_name = feat.replace("Griffin_text_", "")
                if origin_name in nodelist[nodetype]['feat']:
                    nodelist[nodetype]['feat'].remove(origin_name)
                    featlist.pop(origin_name)
    for nodetype in nodelist:
        featnamelist = copy.deepcopy(nodelist[nodetype]['feat'])
        nodelist[nodetype]["is_target"] = False
        for feat in featnamelist:
            if featlist[feat][1][0]:
                assert len(featnamelist) == 1 # target have only one target column
                nodelist[nodetype]["is_target"] = True
                print("target node", nodetype)
    return nodelist, featlist

nodelist, featlist = buildnodemeta(config)
with open(osp.join(dstpath, "metanode.yaml"), "w") as f:
    tnodelist = copy.deepcopy(nodelist)
    for nodetype in tnodelist:
        tnodelist[nodetype].pop("timestamp")
        # tnodelist[nodetype]["feat"].append("timestamp")
    yaml.dump(tnodelist, f)


torch.save({featname: torch.from_numpy(featlist[featname][-1]).to(torch.float32) for featname in featlist}, osp.join(dstpath, "featnameemb.pt"))

from datasets import Dataset

def process(nodetype):
    featembs = {}
    timestamp = np.load(osp.join(srcpath, nodelist[nodetype]["timestamp"]))
    if nodelist[nodetype]["is_target"]:
        assert np.all(timestamp==0)
        timestamp[:] = -9223372036854775808 # -inf for int64
    elif np.all(timestamp==0):
        timestamp[:] = -9223372036854775808 # -inf for int64
    nodefeatdict = {"timestamp": timestamp}
    #assert np.all(np.diff(nodefeatdict["timestamp"]) <= 0), False
    textfeat_compressed = None
    for featname in nodelist[nodetype]["feat"]:
        path, extra, featemb = featlist[featname]
        is_target_column, dtype, num_categories = extra
        
        featembs[featname] = torch.from_numpy(featemb)

        feattensor = np.load(osp.join(srcpath, path))
        if "Griffin_text_" in featname:
            ufeattensor, feattensor = torch.unique(torch.from_numpy(feattensor), dim=0, return_inverse=True)
            if textfeat_compressed is None:
                textfeat_compressed = ufeattensor
            else:
                feattensor += textfeat_compressed.shape[0]
                textfeat_compressed = torch.concat((textfeat_compressed, ufeattensor), dim=0)
            print(featname, " compress to ", ufeattensor.shape[0], feattensor.shape[0])
            #assert ufeattensor.shape[0] < feattensor.shape[0]//10, False
        
        nodefeatdict[featname] = feattensor

    ds = Dataset.from_dict(nodefeatdict)
    ds.save_to_disk(osp.join(dstpath, "node/", nodetype, "feat"))
    ds = Dataset.from_dict({"emb": textfeat_compressed})
    ds.save_to_disk(osp.join(dstpath, "node/", nodetype, "textemb"))

from pqdm.processes import pqdm
pqdm(list(nodelist.keys()), process, n_jobs=args.ncpu)

exit()

#print(config.keys())#['dataset_name', 'feature_data', 'graph', 'tasks']
#print(config["dataset_name"]) #str, Joint-v5-r2n-griffin
    
# print(config["feature_data"]) a list
# print(config["feature_data"][0].keys()) # ['domain', 'extra_fields', 'format', 'in_memory', 'name', 'path', 'type']
# print([_["domain"] for _ in config["feature_data"]]) # always node
# print(config["feature_data"][0]["extra_fields"]) # {'dtype': 'category', 'name_emb': floatlist, 'num_categories': 4}
# print(config["feature_data"][0]["format"]) # always numpy
# print(config["feature_data"][0]["in_memory"]) # always False
# print(config["feature_data"][0]["name"]) # platform
# print(config["feature_data"][0]["path"]) # features/outbrain-small-Event_platform.npy
# print(config["feature_data"][0]["type"]) # node type outbrain-small-Event

# print(config["graph"].keys()) # dict_keys(['edges', 'feature_data', 'nodes'])
# print(config["graph"]["nodes"]) # list of {'num': 109972, 'type': 'outbrain-small-Event'}
# print(config["graph"]["edges"]) # list of {'format': 'numpy', 'path': 'edges/retailrocket-View_added_to_cart:retailrocket-View_added_to_cart-self_loop:retailrocket-View_added_to_cart_edges.npy', 'type': 'retailrocket-View_added_to_cart:retailrocket-View_added_to_cart-self_loop:retailrocket-View_added_to_cart'}, no edge type information
# print(config["graph"]["feature_data"]) # all time stamp. list of {'domain': 'node', 'extra_fields': {}, 'format': 'numpy', 'in_memory': False, 'name': '__timestamp__', 'path': 'features/outbrain-small-Event___timestamp__.npy', 'type': 'outbrain-small-Event'}
# all domains are nodes, each is a node's time stamp

# print(config["tasks"]) # list
# print(config["tasks"][0].keys()) #'extra_fields', 'test_set', 'train_set', 'validation_set'