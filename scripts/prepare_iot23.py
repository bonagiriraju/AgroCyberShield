import argparse
from agrocybershield.data import load_source
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--timestamp-col'); p.add_argument('--label-col'); p.add_argument('--group-col'); a=p.parse_args(); d=load_source(a.input,'iot23',a.timestamp_col,a.label_col,a.group_col); d.to_csv(a.output,index=False); print(d.shape,a.output)
