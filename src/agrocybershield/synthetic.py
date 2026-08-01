import numpy as np,pandas as pd
from .data import CLASSES
def generate_source(source='ton_iot',groups=8,rows_per_group=240,seed=42):
 rng=np.random.default_rng(seed+(0 if source=='ton_iot' else 1)); rows=[]
 for g in range(groups):
  for t in range(rows_per_group):
   k=(t//36+g)%5; label=CLASSES[k]; shift=k*1.2
   rows.append({'timestamp':pd.Timestamp('2025-01-01')+pd.Timedelta(seconds=g*100000+t),'group_id':f'{source}_{g}','source_dataset':source,'original_label':label.lower(),'unified_label':label,'duration':abs(rng.normal(2+shift,.6)),'packets':abs(rng.normal(20+8*shift,5)),'bytes':abs(rng.normal(1000+400*shift,200)),'iat_mean':abs(rng.normal(.5+shift/10,.1)),'rate':abs(rng.normal(10+shift,2)),'protocol':['tcp','udp','mqtt'][t%3],'state':['ok','syn','rst'][k%3]})
 return pd.DataFrame(rows)
