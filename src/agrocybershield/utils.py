from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
import yaml

def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load_yaml(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p

def save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,default=str)

def resolve_device(name='auto'):
    if name=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)
