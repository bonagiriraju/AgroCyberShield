import argparse,joblib,torch
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.training import make_loaders,predict
from agrocybershield.evaluation import evaluate_arrays,save_confusion
from agrocybershield.utils import resolve_device,save_json
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',default='outputs/metrics/test_metrics.json'); a=p.parse_args(); d=joblib.load(a.dataset); dev=resolve_device(); m=FarmSecureNet(d['input_dim'],d['num_classes']).to(dev); m.load_state_dict(torch.load(a.checkpoint,map_location=dev)['model_state']); y,pred,probs,att=predict(m,make_loaders(d)['test'],dev); r=evaluate_arrays(y,pred,probs,d['classes']); save_json(r,a.output); save_confusion(r['confusion_matrix'],d['classes'],'outputs/figures/confusion_matrix.png'); print(r['accuracy'],r['macro_f1'])
