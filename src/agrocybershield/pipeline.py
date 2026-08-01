from __future__ import annotations
import joblib, numpy as np, pandas as pd
from pathlib import Path
from .data import CLASSES,temporal_split,make_sequences
from .preprocessing import AgroPreprocessor

def build_dataset(ton_df,iot_df,output,length=20,stride=5,use_pca=True):
    all_df=pd.concat([ton_df,iot_df],ignore_index=True,sort=False); all_df=temporal_split(all_df)
    label_to_id={c:i for i,c in enumerate(CLASSES)}; train=all_df[all_df.partition=='train'].copy(); y=train.unified_label.map(label_to_id).to_numpy()
    pp=AgroPreprocessor(use_pca=use_pca).fit(train,y)
    transformed=[]
    for part in ['train','val','test']:
        d=all_df[all_df.partition==part].copy(); A=pp.transform(d)
        z=pd.DataFrame(A,columns=pp.feature_names_); meta=d[['timestamp','group_id','source_dataset','original_label','unified_label','partition']].reset_index(drop=True); transformed.append(pd.concat([meta,z],axis=1))
    frame=pd.concat(transformed,ignore_index=True); data={'classes':CLASSES,'num_classes':len(CLASSES),'feature_names':pp.feature_names_,'input_dim':len(pp.feature_names_),'preprocessor':pp}
    for part in ['train','val','test']:
        d=frame[frame.partition==part]; X,Y,M=make_sequences(d,pp.feature_names_,label_to_id,length,stride); data[part]={'X':X,'y':Y,'meta':M}
    Path(output).parent.mkdir(parents=True,exist_ok=True); joblib.dump(data,output); return data
