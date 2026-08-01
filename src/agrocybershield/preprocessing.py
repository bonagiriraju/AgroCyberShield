from __future__ import annotations
from dataclasses import dataclass
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

META={'timestamp','group_id','source_dataset','original_label','unified_label','partition'}
ID_RE=re.compile(r'(^|_)(ip|mac|address|id|uid|uuid|src_port|dst_port|source_port|destination_port)($|_)',re.I)

@dataclass
class AgroPreprocessor:
    corr_threshold:float=0.95
    pca_variance:float=0.95
    upper_quantile:float|None=0.99
    lower_quantile:float|None=None
    use_pca:bool=True
    def fit(self,df,y):
        raw=[c for c in df.columns if c not in META and not ID_RE.search(str(c))]
        self.raw_features_=raw
        self.numeric_=[c for c in raw if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_=[c for c in raw if c not in self.numeric_]
        transformers=[]
        if self.numeric_:
            transformers.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),self.numeric_))
        if self.categorical_:
            transformers.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),self.categorical_))
        self.ct_=ColumnTransformer(transformers,remainder='drop',verbose_feature_names_out=True)
        self.bounds_={}
        for c in self.numeric_:
            lo=df[c].quantile(self.lower_quantile) if self.lower_quantile is not None else None
            hi=df[c].quantile(self.upper_quantile) if self.upper_quantile is not None else None
            self.bounds_[c]=(lo,hi)
        z=self._clip(df); A=self.ct_.fit_transform(z); names=np.asarray(self.ct_.get_feature_names_out())
        var=np.nanvar(A,axis=0); keep=var>1e-12; A=A[:,keep]; names=names[keep]
        self.variance_keep_=keep
        if A.shape[1]>1:
            corr=np.corrcoef(A,rowvar=False); drop=set()
            try: mi=mutual_info_classif(A,y,random_state=42)
            except Exception: mi=np.zeros(A.shape[1])
            for i in range(A.shape[1]):
                for j in range(i+1,A.shape[1]):
                    if j not in drop and np.isfinite(corr[i,j]) and abs(corr[i,j])>self.corr_threshold:
                        drop.add(i if mi[i]<mi[j] else j)
            self.corr_keep_=np.array([i not in drop for i in range(A.shape[1])]); A=A[:,self.corr_keep_]; names=names[self.corr_keep_]
        else: self.corr_keep_=np.ones(A.shape[1],dtype=bool)
        self.pre_pca_names_=list(map(str,names))
        if self.use_pca and A.shape[1]>1:
            self.pca_=PCA(n_components=self.pca_variance,svd_solver='full',random_state=42).fit(A)
            self.feature_names_=[f'pc_{i+1}' for i in range(self.pca_.n_components_)]
        else: self.pca_=None; self.feature_names_=self.pre_pca_names_
        return self
    def _clip(self,df):
        z=df.copy()
        for c,(lo,hi) in getattr(self,'bounds_',{}).items():
            if c in z:
                z[c]=pd.to_numeric(z[c],errors='coerce')
                if lo is not None: z[c]=z[c].clip(lower=lo)
                if hi is not None: z[c]=z[c].clip(upper=hi)
        return z
    def transform(self,df):
        A=self.ct_.transform(self._clip(df)); A=A[:,self.variance_keep_]; A=A[:,self.corr_keep_]
        if self.pca_ is not None: A=self.pca_.transform(A)
        return np.asarray(A,dtype=np.float32)
    def save(self,path): joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
