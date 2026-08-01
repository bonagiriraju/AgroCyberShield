from __future__ import annotations
import torch
from torch import nn

class AdditiveAttention(nn.Module):
    def __init__(self,input_dim,attention_dim):
        super().__init__(); self.proj=nn.Linear(input_dim,attention_dim); self.score=nn.Linear(attention_dim,1,bias=False)
    def forward(self,x,mask=None):
        e=self.score(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None: e=e.masked_fill(~mask,-1e9)
        a=torch.softmax(e,dim=1); c=torch.sum(x*a.unsqueeze(-1),dim=1); return c,a

class FarmSecureNet(nn.Module):
    def __init__(self,input_dim,num_classes=5,conv_channels=(64,128),kernels=(5,3),lstm_hidden=128,attention_dim=128,dense_dims=(128,64),dropout=.3,use_cnn=True,use_bilstm=True,use_attention=True):
        super().__init__(); self.use_cnn=use_cnn; self.use_bilstm=use_bilstm; self.use_attention=use_attention
        d=input_dim
        if use_cnn:
            layers=[]
            for i,(ch,k) in enumerate(zip(conv_channels,kernels)):
                layers += [nn.Conv1d(d,ch,k,padding=k//2),nn.BatchNorm1d(ch),nn.ReLU(),nn.Dropout(.1)]
                d=ch
            self.cnn=nn.Sequential(*layers); self.pool=nn.MaxPool1d(2)
        if use_bilstm:
            self.rnn=nn.LSTM(d,lstm_hidden,batch_first=True,bidirectional=True); d=2*lstm_hidden
        if use_attention: self.attention=AdditiveAttention(d,attention_dim)
        mlp=[]; last=d
        for h in dense_dims: mlp += [nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)]; last=h
        self.classifier=nn.Sequential(*mlp,nn.Linear(last,num_classes))
    def forward(self,x):
        if self.use_cnn:
            x=self.cnn(x.transpose(1,2)); x=self.pool(x).transpose(1,2)
        if self.use_bilstm: x,_=self.rnn(x)
        if self.use_attention: z,a=self.attention(x)
        else: z=x.mean(dim=1); a=torch.full((x.size(0),x.size(1)),1/x.size(1),device=x.device)
        return self.classifier(z),a
