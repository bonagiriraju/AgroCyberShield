from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix,roc_auc_score
from .utils import ensure_dir,save_json

def evaluate_arrays(y,p,probs,class_names):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0); wpr,wrc,wf1,_=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)
    out={'accuracy':accuracy_score(y,p),'macro_precision':pr,'macro_recall':rc,'macro_f1':f1,'weighted_precision':wpr,'weighted_recall':wrc,'weighted_f1':wf1,'classification_report':classification_report(y,p,target_names=class_names,labels=list(range(len(class_names))),output_dict=True,zero_division=0)}
    try: out['macro_ovr_auc']=roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=list(range(len(class_names))))
    except Exception: out['macro_ovr_auc']=None
    out['confusion_matrix']=confusion_matrix(y,p,labels=list(range(len(class_names)))).tolist(); return out

def save_confusion(cm,class_names,path):
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm); ax.set_xticks(range(len(class_names)),class_names,rotation=45,ha='right'); ax.set_yticks(range(len(class_names)),class_names); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(len(class_names)):
      for j in range(len(class_names)): ax.text(j,i,str(cm[i][j]),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=300); plt.close(fig)
