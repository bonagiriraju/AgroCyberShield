import argparse,pandas as pd
from agrocybershield.pipeline import build_dataset
p=argparse.ArgumentParser(); p.add_argument('--ton',required=True); p.add_argument('--iot23',required=True); p.add_argument('--output',default='data/processed/dataset.joblib'); p.add_argument('--length',type=int,default=20); p.add_argument('--stride',type=int,default=5); a=p.parse_args(); d=build_dataset(pd.read_csv(a.ton),pd.read_csv(a.iot23),a.output,a.length,a.stride); print({k:d[k]['X'].shape for k in ['train','val','test']})
