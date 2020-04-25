import torch
import torch.nn as nn
import numpy as np
import torch.jit
from model.multi_layer_cnn import MultiLayerCNN
from model.decoder import Decoder
from model.attention import SelfAttention, SelfAttention_fate
# from patent.model.multi_layer_cnn import MultiLayerCNN
# from patent.model.decoder import Decoder
# from patent.model.attention import SelfAttention, SelfAttention_fate


class IMN(nn.Module):
    def __init__(self, gen_emb, domain_emb, ae_nums, as_nums, ds_nums, iters=15, dropout=0.5, use_transission=True):
        """
        :param gen_emb: 通用词向量权重
        :param domain_emb: 领域词向量权重
        :param ae_nums: aspect和opinion word抽取的标签种类
        :param as_nums: aspect sentiment种类
        :param ds_nums: doc sentiment种类
        :param iters: message passing轮数
        :param dropout:
        :param use_opinion: AE和AS之间是否建立联系
        """
        super(IMN, self).__init__()
        self.iters = iters
        self.use_transission = use_transission
        self.dropout = nn.Dropout(dropout)
        # f_s
        self.general_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.general_embedding.weight = torch.nn.Parameter(gen_emb, requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(domain_emb, requires_grad=False)
        self.encoder_shared = MultiLayerCNN(400, 2, dropout=dropout)

        # f_ae
        self.encoder_aspect = MultiLayerCNN(256, 2, dropout=dropout)
        self.decoder_aspect = Decoder(912, ae_nums)

        # f_as
        # self.att_sentiment = SelfAttention(256, use_transission=self.use_transission)
        self.att_sentiment = SelfAttention_fate(256)
        self.encoder_sentiment = MultiLayerCNN(256, 0, dropout=dropout)
        self.decoder_sentiment = Decoder(512, as_nums)
        # update
        self.update = nn.Linear(256 + as_nums + ae_nums, 256)

    def emb(self, features):
        general_features = self.general_embedding(features)
        domain_features = self.domain_embedding(features)
        features = torch.cat((general_features, domain_features), dim=2)
        return features

    def forward(self, feature, op_label_feature=None,ap_label_feature=None, p_gold_op=None, mask=None):
        """
        :param feature: batch,sent_len
        :param op_label_feature: batch,sent_len,3
        :param p_gold_op: batch,sent_len
        :param mask: batch,sent_len
        :return:
        """
        # sentiment shared layers
        feature_emb = self.emb(feature)  # batch,sent_len,domain+general=400
        feature_shared = self.encoder_shared(feature_emb)  # batch,sent_len,256
        feature_emb = torch.cat((feature_emb, feature_shared), dim=-1)  # batch,sent_len,656
        init_shared_features = feature_shared  # batch,sent_len,256

        aspect_probs = None
        sentiment_probs = None
        # task specfic layers
        for i in range(self.iters):
            aspect_output = feature_shared
            sentiment_output = feature_shared
            ### AE ###
            aspect_output = self.encoder_aspect(aspect_output)  # batch,sent_len,256
            aspect_output = torch.cat((feature_emb, aspect_output), dim=-1)  # batch,sent_len,656+256
            aspect_output = self.dropout(aspect_output)
            aspect_probs = self.decoder_aspect(aspect_output)
            ### AS ###
            sentiment_output = self.encoder_sentiment(sentiment_output)  # batch,sent_len,256
            sentiment_output = self.att_sentiment(sentiment_output, op_label_feature,ap_label_feature, aspect_probs,
                                                  p_gold_op)  # batch,sent_len,256
            sentiment_output = torch.cat((init_shared_features, sentiment_output), dim=-1)  # batch,sent_len,256+256
            sentiment_output = self.dropout(sentiment_output)
            sentiment_probs = self.decoder_sentiment(sentiment_output)
            feature_shared = torch.cat((feature_shared, aspect_probs, sentiment_probs), dim=-1)
            feature_shared = self.update(feature_shared)

        aspect_probs = torch.log(aspect_probs)
        sentiment_probs = torch.log(sentiment_probs)

        return aspect_probs, sentiment_probs

class NewImn(nn.Module):
    def __init__(self, gen_emb, domain_emb, ae_nums, as_nums, ds_nums, iters=15, dropout=0.5, use_transission=True):
        """
        :param gen_emb: 通用词向量权重
        :param domain_emb: 领域词向量权重
        :param ae_nums: aspect和opinion word抽取的标签种类
        :param as_nums: aspect sentiment种类
        :param ds_nums: doc sentiment种类
        :param iters: message passing轮数
        :param dropout:
        :param use_opinion: AE和AS之间是否建立联系
        """
        super(NewImn, self).__init__()
        self.iters = iters
        self.use_transission = use_transission
        self.dropout = nn.Dropout(dropout)
        # f_s
        self.general_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.general_embedding.weight = torch.nn.Parameter(gen_emb, requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(domain_emb, requires_grad=False)
        self.encoder_shared = MultiLayerCNN(400, 0, dropout=dropout)

        # f_ae
        self.encoder_aspect = nn.GRU(256,128,num_layers=1,batch_first=True,bidirectional=True)
        # self.encoder_aspect = MultiLayerCNN(256,2,dropout=dropout)
        self.decoder_aspect = Decoder(912, ae_nums)

        # f_as
        self.encoder_sentiment = nn.GRU(661,256,num_layers=1,batch_first=True,bidirectional=True)
        self.decoder_sentiment = Decoder(768, as_nums)
        # update
        self.update = nn.Linear(256 + as_nums + ae_nums, 256)

    def emb(self, features):
        general_features = self.general_embedding(features)
        domain_features = self.domain_embedding(features)
        features = torch.cat((general_features, domain_features), dim=2)
        return features

    def forward(self, feature, op_label_feature=None,ap_label_feature=None, p_gold_op=None, mask=None):
        """
        :param feature: batch,sent_len
        :param op_label_feature: batch,sent_len,3
        :param p_gold_op: batch,sent_len
        :param mask: batch,sent_len
        :return:
        """
        # sentiment shared layers
        feature_emb_int = self.emb(feature)  # batch,sent_len,domain+general=400
        feature_shared = self.encoder_shared(feature_emb_int)  # batch,sent_len,256
        feature_emb = torch.cat((feature_emb_int, feature_shared), dim=-1)  # batch,sent_len,656
        init_shared_features = feature_shared  # batch,sent_len,256

        aspect_probs = None
        sentiment_probs = None
        # task specfic layers
        for i in range(self.iters):
            aspect_output = feature_shared
            sentiment_output = feature_shared
            ### AE ###
            aspect_output,_= self.encoder_aspect(aspect_output)  # batch,sent_len,256

            aspect_output = torch.cat((feature_emb, aspect_output), dim=-1)  # batch,sent_len,656+256
            aspect_output = self.dropout(aspect_output)
            aspect_probs = self.decoder_aspect(aspect_output)

            ### AS ###
            sentiment_output = torch.cat((feature_emb_int,sentiment_output,aspect_probs),dim=-1)
            sentiment_output,_ = self.encoder_sentiment(sentiment_output)  # batch,sent_len,512

            sentiment_output = torch.cat((init_shared_features, sentiment_output), dim=-1)  # batch,sent_len,256+512
            sentiment_output = self.dropout(sentiment_output)
            sentiment_probs = self.decoder_sentiment(sentiment_output)

            feature_shared = torch.cat((feature_shared, aspect_probs, sentiment_probs), dim=-1)
            feature_shared = self.update(feature_shared)


        aspect_probs = torch.log(aspect_probs)

        sentiment_probs = torch.log(sentiment_probs)


        return aspect_probs, sentiment_probs

if __name__ == '__main__':
    gen_emb = torch.randn(400, 300)
    domain_emb = torch.randn(400, 100)
    ae_nums, as_nums, ds_nums, dd_nums = 5, 5, 3, 1
    imn = NewImn(gen_emb, domain_emb, ae_nums, as_nums, ds_nums, iters=2)
    feature = torch.LongTensor([2, 3, 4, 1, 1, 0]).unsqueeze(0).repeat(3, 1)
    op_label_feature = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).repeat(3,
                                                                                                                      1,
                                                                                                                      1)
    ap_label_feature = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).repeat(3,
                                                                                                                      1,
                                                                                                                      1)
    mask = None
    p_gold_op = torch.FloatTensor([[1, 0, 1, 1, 0, 1]]).repeat(3, 1)
    aspect_probs, sentiment_probs = imn(feature, op_label_feature,ap_label_feature, p_gold_op, mask)
