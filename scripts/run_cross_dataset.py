import argparse,joblib,numpy as np
from copy import deepcopy
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.training import make_loaders,train_model,predict
from agrocybershield.evaluation import evaluate_arrays
from agrocybershield.utils import resolve_device,save_json
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--epochs',type=int,default=10); a=p.parse_args(); d=joblib.load(a.dataset); results={}; dev=resolve_device()
# Uses stored sequence metadata to select source-specific windows.
for train_src,test_src in [('ton_iot','iot23'),('iot23','ton_iot')]:
 dd=deepcopy(d)
 for part,src in [('train',train_src),('val',train_src),('test',test_src)]:
  mask=np.array([m['source_dataset']==src for m in d[part]['meta']]); dd[part]['X']=d[part]['X'][mask]; dd[part]['y']=d[part]['y'][mask]; dd[part]['meta']=[m for m,k in zip(d[part]['meta'],mask) if k]
 loaders=make_loaders(dd); m=FarmSecureNet(d['input_dim'],d['num_classes']); m,_,_=train_model(m,loaders,dev,epochs=a.epochs,patience=3,checkpoint=f'outputs/checkpoints/cross_{train_src}.pt'); y,pd,pr,_=predict(m,loaders['test'],dev); results[f'{train_src}_to_{test_src}']=evaluate_arrays(y,pd,pr,d['classes'])
save_json(results,'outputs/metrics/cross_dataset.json'); print({k:v['macro_f1'] for k,v in results.items()})
