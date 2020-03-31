import torch
import time
import os
import torch.nn as nn


class UIModel(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(UIModel, self).__init__()
        self.model_name = 'UIModel'
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, use_encoder_final=False):
        """

        :param use_encoder_final: 是否使用encoder隐状态初始化decoder
        :param src: 输入图像数据 （N, C, H, W)
        :param tgt: 对应label2id序列 （N, L)
        :return:
            * scores: (tgt_len * N, tag_vocab_size)
        """
        dec_in = tgt.transpose(0, 1)
        enc_state, context = self.encoder(src)
        self.decoder.init_state(enc_state, use_encoder_final=use_encoder_final)
        dec_out, attn = self.decoder(dec_in, context)
        scores = self.generator(dec_out.contiguous().view(-1, dec_out.size(2)))

        return scores

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
