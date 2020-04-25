"""
IMN 里用的有点多，所以包装一下，名字也跟论文里一样叫decoder了，史上最low decoder
"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self,input_size,tag_nums):
        super(Decoder,self).__init__()
        self.dense = nn.Linear(input_size,tag_nums)

    def forward(self,x):
        x_logit = self.dense(x)
        y = torch.softmax(x_logit,dim=-1)
        return y