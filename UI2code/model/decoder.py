import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, bidirectional_encoder, input_size, num_layers,
                 hidden_size, dropout, vocab_size, attn_type='general', input_feed=False):
        super(RNNDecoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.input_feed = input_feed

        self.embeddings = nn.Embedding(vocab_size, input_size, padding_idx=1)

        # Decoder state
        self.state = {}

        if not self.input_feed:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout)
        else:
            self.rnn = StackedLSTM(num_layers, input_size + hidden_size, hidden_size, dropout)
        # Set up the attention
        self.attn = GlobalAttention(hidden_size, attn_type=attn_type)

    def init_state(self, encoder_final=None, use_encoder_final=False):
        """用encoder的最后一个隐状态来初始化decoder的状态"""

        def _fix_enc_hidden(hidden):
            # (layers*directions) x batch x dim
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            # layers x batch x (dim*directions)
            return hidden

        batch_size = encoder_final[0].size(1)
        h_size = (self.num_layers, batch_size, self.hidden_size)
        feed_size = (1, batch_size, self.hidden_size)
        if use_encoder_final:
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid) for enc_hid in encoder_final)
        else:
            self.state["hidden"] = tuple(enc_hid.data.new(*h_size).zero_()
                                         for enc_hid in encoder_final)
        self.state["input_feed"] = encoder_final[0].data.new(*feed_size).zero_()

    def map_state(self, fn):
        self.state['hidden'] = tuple(fn(h, 1) for h in self.state['hidden'])
        self.state['input_feed'] = fn(self.state['input_feed'], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, context):
        """

        :param tgt: 输入label2id序列 (tgt_len, N)
        :param context: encoder输出 (src_len, N, hidden)
        :return:
            * dec_outs: decoder输出(加入注意力之后） (tgt_len, N, hidden)
            * attn： 注意力分布 (tgt_len, N, src_len)
        """
        emb = self.embeddings(tgt)
        if not self.input_feed:
            rnn_output, dec_state = self.rnn(emb, self.state['hidden'])
            dec_outs, p_attn = self.attn(rnn_output.transpose(0, 1).contiguous(),
                                         context.transpose(0, 1))

            attn = p_attn
            dec_outs = self.dropout(dec_outs)
        else:
            input_feed = self.state['input_feed'].squeeze(0)
            dec_state = self.state['hidden']
            dec_outs = []
            attn = []

            for emb_t in emb.split(1):
                decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
                rnn_output, dec_state = self.rnn(decoder_input, dec_state)
                decoder_output, p_attn = self.attn(rnn_output, context.transpose(0, 1))
                decoder_output = self.dropout(decoder_output)
                input_feed = decoder_output

                dec_outs.append(decoder_output)
                attn.append(p_attn)

            dec_outs = torch.stack(dec_outs)
            attn = torch.stack(attn)

        self.state['hidden'] = dec_state
        self.state['input_feed'] = dec_outs[-1].unsqueeze(0)
        return dec_outs, attn

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        assert attn_type in ['dot', 'general'], (
            "Only support attention type 'dot', 'general' "
            "(got {:s}".format(attn_type))
        self.attn_type = attn_type

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def score(self, h_t, h_s):
        """

        :param h_t: decoder的隐状态 (N, tgt_len, dim)
        :param h_s: 所有encoder隐状态 (N, src_len, dim)
        :return scores (N, tgt_len, src_len)
        """

        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type == 'general':
            h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
            h_t_ = self.linear_in(h_t_)
            h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)

    def forward(self, query, context):
        """

        :param query: 查询向量，也就是decoder的隐状态序列 （N, tgt_len, dim)
        :param context: 信息向量，也就是encoder的所有隐状态 (N, src_len, dim)
        :return:
            * decoder的输出 （tgt_len, N, dim)
            * 注意力分布 （tgt_len, N, src_len)
        """
        if query.dim() == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        batch, target_l, dim = query.size()
        align = self.score(query, context)
        align_vectors = F.softmax(align, -1)
        # 计算最终attention值
        c = torch.bmm(align_vectors, context)
        # 更新decoder隐状态
        concat_c = torch.cat([c, query], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1)
            align_vectors = align_vectors.transpose(0, 1)
        return attn_h, align_vectors


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            # 除了最后一层，都加dropout
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)
