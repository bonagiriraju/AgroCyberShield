import argparse,joblib,csv
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.training import make_loaders,train_model,predict
from agrocybershield.evaluation import evaluate_arrays
from agrocybershield.utils import resolve_device,ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--epochs',type=int,default=10); a=p.parse_args(); d=joblib.load(a.dataset); dev=resolve_device(); variants={'full':{},'no_attention':{'use_attention':False},'no_bilstm':{'use_bilstm':False},'no_cnn':{'use_cnn':False}}; rows=[]
for name,kw in variants.items():
 m=FarmSecureNet(d['input_dim'],d['num_classes'],**kw); loaders=make_loaders(d); m,_,_=train_model(m,loaders,dev,epochs=a.epochs,patience=3,checkpoint=f'outputs/checkpoints/{name}.pt'); y,pd,pr,_=predict(m,loaders['test'],dev); r=evaluate_arrays(y,pd,pr,d['classes']); rows.append({'variant':name,'accuracy':r['accuracy'],'macro_f1':r['macro_f1']})
ensure_dir('outputs/metrics');
with open('outputs/metrics/ablation.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(rows)
