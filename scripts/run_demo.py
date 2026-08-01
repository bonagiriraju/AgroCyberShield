import argparse,joblib,torch
from pathlib import Path
from agrocybershield.synthetic import generate_source
from agrocybershield.pipeline import build_dataset
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.training import make_loaders,train_model,predict
from agrocybershield.evaluation import evaluate_arrays,save_confusion
from agrocybershield.explainability import attention_plot
from agrocybershield.deployment import export_torchscript,benchmark_torch
from agrocybershield.utils import resolve_device,seed_everything,save_json
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=2); a=p.parse_args(); seed_everything(); Path('data/processed').mkdir(parents=True,exist_ok=True); d=build_dataset(generate_source('ton_iot'),generate_source('iot23'),'data/processed/demo.joblib'); loaders=make_loaders(d,64); dev=resolve_device(); m=FarmSecureNet(d['input_dim'],d['num_classes']); m,_,s=train_model(m,loaders,dev,epochs=a.epochs,patience=2,checkpoint='outputs/checkpoints/demo.pt'); y,pred,probs,att=predict(m,loaders['test'],dev); r=evaluate_arrays(y,pred,probs,d['classes']); save_json(r,'outputs/metrics/demo_metrics.json'); save_confusion(r['confusion_matrix'],d['classes'],'outputs/figures/demo_confusion.png'); attention_plot(att,'outputs/figures/demo_attention.png'); ex=torch.tensor(d['test']['X'][:1]); export_torchscript(m,ex,'outputs/edge_models/demo.ts'); save_json(benchmark_torch(m,ex,runs=10,warmup=2),'outputs/metrics/demo_benchmark.json'); print({'train':s,'test_accuracy':r['accuracy'],'macro_f1':r['macro_f1']})
