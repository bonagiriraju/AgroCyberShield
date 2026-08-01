from __future__ import annotations
import time,os
from pathlib import Path
import numpy as np,torch
from .utils import save_json

def export_torchscript(model,example,path):
 model.eval().cpu(); traced=torch.jit.trace(model,example.cpu()); Path(path).parent.mkdir(parents=True,exist_ok=True); traced.save(str(path)); return path
def export_onnx(model,example,path):
 model.eval().cpu(); Path(path).parent.mkdir(parents=True,exist_ok=True); torch.onnx.export(model,example.cpu(),path,input_names=['input'],output_names=['logits','attention'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'},'attention':{0:'batch'}},opset_version=17); return path
def benchmark_torch(model,example,runs=100,warmup=20,device='cpu'):
 dev=torch.device(device); model=model.to(dev).eval(); x=example.to(dev)
 with torch.no_grad():
  for _ in range(warmup): model(x)
  vals=[]
  for _ in range(runs):
   if dev.type=='cuda': torch.cuda.synchronize()
   s=time.perf_counter(); model(x)
   if dev.type=='cuda': torch.cuda.synchronize()
   vals.append((time.perf_counter()-s)*1000)
 return {'runs':runs,'mean_ms':float(np.mean(vals)),'median_ms':float(np.median(vals)),'std_ms':float(np.std(vals)),'p95_ms':float(np.percentile(vals,95))}
