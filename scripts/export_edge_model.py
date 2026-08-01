import argparse,joblib,torch
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.deployment import export_torchscript,export_onnx,benchmark_torch
from agrocybershield.utils import save_json
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--checkpoint',required=True); a=p.parse_args(); d=joblib.load(a.dataset); m=FarmSecureNet(d['input_dim'],d['num_classes']); m.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model_state']); x=torch.tensor(d['test']['X'][:1]); export_torchscript(m,x,'outputs/edge_models/farmsecurenet.ts');
try: export_onnx(m,x,'outputs/edge_models/farmsecurenet.onnx')
except Exception as e: print('ONNX export skipped:',e)
save_json(benchmark_torch(m,x),'outputs/metrics/edge_benchmark.json'); print('Export complete')
