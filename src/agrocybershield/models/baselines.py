import torch
from torch import nn
from .farmsecurenet import AdditiveAttention
class CNNBaseline(nn.Module):
 def __init__(self,input_dim,num_classes=5):
  super().__init__(); self.net=nn.Sequential(nn.Conv1d(input_dim,64,5,padding=2),nn.ReLU(),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.fc=nn.Linear(128,num_classes)
 def forward(self,x): return self.fc(self.net(x.transpose(1,2)).squeeze(-1)),None
class BiLSTMBaseline(nn.Module):
 def __init__(self,input_dim,num_classes=5): super().__init__(); self.rnn=nn.LSTM(input_dim,128,batch_first=True,bidirectional=True); self.fc=nn.Linear(256,num_classes)
 def forward(self,x): z,_=self.rnn(x); return self.fc(z.mean(1)),None
class GRUAttention(nn.Module):
 def __init__(self,input_dim,num_classes=5): super().__init__(); self.rnn=nn.GRU(input_dim,128,batch_first=True,bidirectional=True); self.att=AdditiveAttention(256,128); self.fc=nn.Linear(256,num_classes)
 def forward(self,x): z,_=self.rnn(x); c,a=self.att(z); return self.fc(c),a
class TransformerIDS(nn.Module):
 def __init__(self,input_dim,num_classes=5):
  super().__init__(); self.proj=nn.Linear(input_dim,128); layer=nn.TransformerEncoderLayer(128,4,256,batch_first=True); self.enc=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(128,num_classes)
 def forward(self,x): z=self.enc(self.proj(x)); return self.fc(z.mean(1)),None
