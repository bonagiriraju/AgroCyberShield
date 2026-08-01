from __future__ import annotations
from pathlib import Path
import copy,time
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from sklearn.metrics import f1_score
from .data import SequenceDataset
from .utils import ensure_dir

def make_loaders(data,batch_size=64,weighted=True):
    loaders={}
    for part in ['train','val','test']:
        X,y=data[part]['X'],data[part]['y']; ds=SequenceDataset(X,y)
        sampler=None; shuffle=part=='train'
        if part=='train' and weighted and len(y):
            counts=np.bincount(y,minlength=data['num_classes']); weights=1/np.maximum(counts,1); sampler=WeightedRandomSampler(weights[y],len(y),replacement=True); shuffle=False
        loaders[part]=DataLoader(ds,batch_size=batch_size,shuffle=shuffle,sampler=sampler,num_workers=0)
    return loaders

@torch.no_grad()
def predict(model,loader,device):
    model.eval(); ys=[]; ps=[]; probs=[]; att=[]
    for x,y in loader:
        x=x.to(device); logits,a=model(x); p=torch.softmax(logits,1)
        ys.extend(y.numpy()); ps.extend(p.argmax(1).cpu().numpy()); probs.extend(p.cpu().numpy());
        if a is not None: att.extend(a.cpu().numpy())
    return np.asarray(ys),np.asarray(ps),np.asarray(probs),np.asarray(att)

def train_model(model,loaders,device,epochs=100,lr=.001,weight_decay=1e-5,patience=10,gradient_clip=5.0,checkpoint='outputs/checkpoints/farmsecurenet_best.pt'):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay); loss_fn=torch.nn.CrossEntropyLoss()
    best=-1; state=None; wait=0; history=[]; start=time.time()
    for epoch in range(1,epochs+1):
        model.train(); total=0; n=0
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits,_=model(x); loss=loss_fn(logits,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip); opt.step(); total+=loss.item()*len(y); n+=len(y)
        vy,vp,_,_=predict(model,loaders['val'],device); score=f1_score(vy,vp,average='macro',zero_division=0) if len(vy) else 0
        history.append({'epoch':epoch,'train_loss':total/max(n,1),'val_macro_f1':score})
        if score>best+1e-6: best=score; state=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if wait>=patience: break
    if state is not None: model.load_state_dict(state)
    p=Path(checkpoint); ensure_dir(p.parent); torch.save({'model_state':model.state_dict(),'history':history,'best_val_macro_f1':best},p)
    return model,history,{'best_val_macro_f1':best,'seconds':time.time()-start}
