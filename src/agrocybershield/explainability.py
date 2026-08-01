from __future__ import annotations
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt

def attention_plot(att,path):
 a=np.asarray(att); mean=a.mean(0) if a.ndim==2 else a
 fig,ax=plt.subplots(figsize=(7,3)); ax.plot(np.arange(1,len(mean)+1),mean,marker='o'); ax.set_xlabel('Model time step'); ax.set_ylabel('Mean attention'); ax.grid(False); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=300); plt.close(fig)

def gradient_attributions(model,x,target=None):
 model.eval(); x=x.clone().detach().requires_grad_(True); logits,_=model(x); target=logits.argmax(1) if target is None else target; selected=logits.gather(1,target.view(-1,1)).sum(); selected.backward(); return (x.grad*x).detach().cpu().numpy()

def shap_explain(model,background,samples):
 try:
  import shap
  class LogitWrapper(torch.nn.Module):
   def __init__(self,m): super().__init__(); self.m=m
   def forward(self,x): return self.m(x)[0]
  return shap.GradientExplainer(LogitWrapper(model),background).shap_values(samples)
 except Exception as e: raise RuntimeError('SHAP explanation failed. Install optional dependency shap.') from e
