import argparse,joblib,torch,numpy as np
from agrocybershield.models.farmsecurenet import FarmSecureNet
from agrocybershield.explainability import gradient_attributions,attention_plot
from agrocybershield.utils import resolve_device
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--samples',type=int,default=100); a=p.parse_args(); d=joblib.load(a.dataset); dev=resolve_device(); m=FarmSecureNet(d['input_dim'],d['num_classes']).to(dev); m.load_state_dict(torch.load(a.checkpoint,map_location=dev)['model_state']); x=torch.tensor(d['test']['X'][:a.samples],device=dev); attr=gradient_attributions(m,x); np.save('outputs/metrics/gradient_attributions.npy',attr); _,att=m(x); attention_plot(att.detach().cpu().numpy(),'outputs/figures/attention.png'); print(attr.shape)
