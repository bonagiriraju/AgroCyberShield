from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES=['Normal','DoS','Injection','Botnet','Malware']

def _norm(s): return re.sub(r'[^a-z0-9]+','_',str(s).strip().lower()).strip('_')

TON_MAP={
 'normal':'Normal','benign':'Normal','dos':'DoS','ddos':'DoS','injection':'Injection','xss':'Injection',
 'backdoor':'Malware','ransomware':'Malware','scanning':'Malware','scan':'Malware','password':'Malware',
 'brute_force':'Malware','mitm':'Malware','man_in_the_middle':'Malware'}
IOT23_MAP={'benign':'Normal','normal':'Normal','mirai':'Botnet','gafgyt':'Botnet','torii':'Botnet',
 'dos':'DoS','ddos':'DoS','c_c':'Malware','command_and_control':'Malware','scanning':'Malware',
 'scan':'Malware','malicious_download':'Malware','protocol_abuse':'Malware','malware':'Malware'}

def map_label(label, source):
    x=_norm(label)
    table=TON_MAP if source=='ton_iot' else IOT23_MAP if source=='iot23' else {}
    if x in table: return table[x]
    for key,val in table.items():
        if key and key in x: return val
    if any(k in x for k in ['benign','normal']): return 'Normal'
    if 'ddos' in x or re.search(r'(^|_)dos($|_)',x): return 'DoS'
    if any(k in x for k in ['mirai','gafgyt','torii','botnet']): return 'Botnet'
    if any(k in x for k in ['inject','xss','sql']): return 'Injection'
    return 'Malware'

def read_csv_tree(path):
    p=Path(path); fs=[p] if p.is_file() else sorted(p.rglob('*.csv'))
    if not fs: raise FileNotFoundError(f'No CSV files under {p}')
    out=[]
    for f in fs:
        try:
            d=pd.read_csv(f,low_memory=False); d['__source_file__']=f.name; out.append(d)
        except Exception as e: print(f'Skipping {f}: {e}')
    if not out: raise ValueError('No readable CSV files')
    return pd.concat(out,ignore_index=True,sort=False)

def choose_col(df,candidates,required=True):
    mapping={_norm(c):c for c in df.columns}
    for c in candidates:
        if _norm(c) in mapping: return mapping[_norm(c)]
    if required: raise ValueError(f'Could not identify column from {candidates}')
    return None

def load_source(path,source,timestamp_col=None,label_col=None,group_col=None):
    df=read_csv_tree(path)
    timestamp_col=timestamp_col or choose_col(df,['timestamp','ts','start_time','date','time'],False)
    label_col=label_col or choose_col(df,['label','type','attack','category','detailed_label'],True)
    group_col=group_col or choose_col(df,['device_id','device','source','sensor','scenario','capture','uid','file_id'],False)
    out=df.copy()
    if timestamp_col:
        parsed=pd.to_datetime(out[timestamp_col],errors='coerce',utc=True)
        if parsed.notna().sum()>0: out['timestamp']=parsed
        else: out['timestamp']=pd.to_numeric(out[timestamp_col],errors='coerce')
    else: out['timestamp']=np.arange(len(out))
    out['group_id']=out[group_col].astype(str) if group_col else out['__source_file__'].astype(str)
    out['source_dataset']=source
    out['original_label']=out[label_col].astype(str)
    out['unified_label']=out['original_label'].map(lambda x:map_label(x,source))
    drop={label_col,timestamp_col,group_col,'__source_file__','source_dataset','original_label','unified_label','group_id','timestamp',None}
    feats=[c for c in out.columns if c not in drop]
    return out[['timestamp','group_id','source_dataset','original_label','unified_label']+feats]

def temporal_split(df,train=0.8,val=0.1):
    parts=[]
    for (_,g),x in df.groupby(['source_dataset','group_id'],sort=False):
        x=x.sort_values('timestamp').reset_index(drop=True); n=len(x); a=max(1,int(n*train)); b=max(a+1,int(n*(train+val)))
        y=x.copy(); y['partition']='test'; y.loc[:a-1,'partition']='train'; y.loc[a:b-1,'partition']='val'; parts.append(y)
    return pd.concat(parts,ignore_index=True)

def make_sequences(df,feature_cols,label_to_id,length=20,stride=5):
    X=[]; y=[]; meta=[]
    for (src,grp,part),g in df.groupby(['source_dataset','group_id','partition'],sort=False):
        g=g.sort_values('timestamp'); arr=g[feature_cols].to_numpy(np.float32); labs=g['unified_label'].map(label_to_id).to_numpy()
        for s in range(0,len(g)-length+1,stride):
            X.append(arr[s:s+length]); y.append(int(labs[s+length-1])); meta.append({'source_dataset':src,'group_id':grp,'partition':part,'end_timestamp':str(g.iloc[s+length-1]['timestamp'])})
    if not X: return np.empty((0,length,len(feature_cols)),np.float32),np.empty((0,),np.int64),[]
    return np.stack(X),np.asarray(y,np.int64),meta

class SequenceDataset(Dataset):
    def __init__(self,X,y): self.X=torch.as_tensor(X,dtype=torch.float32); self.y=torch.as_tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i],self.y[i]
