import argparse,joblib
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.training import make_loaders,train_model
from agrocybershield.utils import resolve_device,seed_everything
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--checkpoint',default='outputs/checkpoints/farmsecurenet_best.pt'); a=p.parse_args(); seed_everything(); d=joblib.load(a.dataset); loaders=make_loaders(d,a.batch_size); m=FarmSecureNet(d['input_dim'],d['num_classes']); m,h,s=train_model(m,loaders,resolve_device(),a.epochs,checkpoint=a.checkpoint); print(s)
