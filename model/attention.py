import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
epsilon = 1e-5
"""
缺少mask
带有距离信息的self-attention
"""
class SelfAttention(nn.Module):
    def __init__(self, input_size,use_transission):
        super(SelfAttention, self).__init__()
        self.use_transission = use_transission
        self.W = nn.Linear(input_size, input_size)

    def forward(self, x,gold_opinion,gold_aspect,predict_label,gold_prob):
        device = x.device
        steps = x.size(-2)
        x_tran = self.W(x)
        x_transpose = x_tran.permute(0, 2, 1)
        weights = torch.matmul(x_tran, x_transpose)
        location = np.abs(
            np.tile(np.array(range(steps)), (steps, 1)) - np.array(range(steps)).reshape(steps, 1))
        location = torch.FloatTensor(location).to(device)
        loc_weights = 1.0/(location+epsilon)
        weights = weights * loc_weights

        if self.use_transission:
            #gold_opinion:0,1,2:O,BP，IP
            #predict_opinion:0,1,2,3,4:O,BA,IA,BP,IP
            gold_opinion_ = gold_opinion[:,:,1]+gold_opinion[:,:,2]
            gold_aspect_ = gold_aspect[:,:,1]+gold_aspect[:,:,2]

            predict_label_ = predict_label[:,:,1]+predict_label[:,:,2] + predict_label[:,:,3] + predict_label[:,:,4]
            opinion_weights = gold_prob*(gold_opinion_+ gold_aspect_) + (1-gold_prob)*predict_label_
            opinion_weights = opinion_weights.unsqueeze(-1)
            weights = weights*opinion_weights

        weights = torch.tanh(weights)
        weights = weights.exp()
        weights = weights * torch.Tensor(np.eye(steps) == 0).to(device)
        weights = weights/(torch.sum(weights,dim=-1,keepdim=True)+epsilon)
        return torch.matmul(weights,x)

class SelfAttention_fate(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention_fate, self).__init__()
        self.W1 = nn.Linear(input_size, input_size)
        self.W2 = nn.Linear(input_size,input_size)
        nn.init.normal_(self.W1.weight)
        nn.init.normal_(self.W2.weight)

    def forward(self, x,gold_opinion,gold_aspect,predict_label,gold_prob):
        device = x.device
        steps = x.size(-2)

        gold_opinion_ = gold_opinion[:, :, 1] + gold_opinion[:, :, 2]
        gold_aspect_ = gold_aspect[:, :, 1] + gold_aspect[:, :, 2]
        predict_opinion_ = predict_label[:, :, 3] + predict_label[:, :, 4]
        predict_aspect_ = predict_label[:, :, 1] + predict_label[:, :, 2]

        opinion_weights = gold_prob*gold_opinion_ + (1-gold_prob)*predict_opinion_
        aspect_weights = gold_prob*gold_aspect_ + (1-gold_prob)*predict_aspect_

        print(opinion_weights)
        return opinion_weights

class DotSentimentAttention(nn.Module):
    def __init__(self,input_size):
        super(DotSentimentAttention,self).__init__()
        self.W = nn.Parameter(torch.Tensor(input_size,))
        nn.init.normal_(self.W.data)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        eij = torch.sum(x*self.W,dim=-1)+self.bias
        a_exp = eij.exp()
        a_sigmoid = torch.sigmoid(a_exp)
        a_softmax = torch.softmax(eij,dim=-1)
        return a_softmax,a_sigmoid


if __name__ == '__main__':
    Att = SelfAttention_fate(5)
    a = torch.randn(3,6,5)
    op_label_feature = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).repeat(3,
                                                                                                                      1,
                                                                                                                      1)
    ap_label_feature = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).repeat(3,
                                                                                                                      1,
                                                                                                                      1)
    p_gold_op = torch.FloatTensor([[1, 1, 1, 1, 1, 1]]).repeat(3, 1)

    predict = torch.softmax(torch.randn(3,6,5).float(),dim=-1)
    b = Att(a,op_label_feature,ap_label_feature,predict,p_gold_op)


