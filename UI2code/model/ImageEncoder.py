import torch.nn as nn
import torch.nn.functional as F
import torch


class ImageEncoder(nn.Module):
    def __init__(self, num_layers, bidirectional, rnn_size, dropout, image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(image_chanel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        dropout = dropout[0] if type(dropout) is list else dropout
        self.rnn = nn.LSTM(input_size=src_size,
                           hidden_size=int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, src_size)

    def forward(self, src):
        """

        :param src: 输入图像数据 （N, C, H, W)
        :return:
            * hidden_t: 最后的隐状态  （[layers*directions, N, hidden_size], [layers*directions, N, hidden_size]）
            * out: encoder输出，也就是所有的隐状态即context (H/2/2/2 * (W/2/2/2 + 1), N, rnn_size)
        """
        batch_size = src.size(0)
        # layer 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), inplace=True)  # (N, 64, H, W)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))  # (N, 64, H/2, W/2)
        # layer 2
        src = F.relu(self.layer2(src), inplace=True)  # (N, 128, H/2, W/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))  # (N, 128, H/2/2, W/2/2)
        # layer 3   batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), inplace=True)  # (N, 256, H/2/2, W/2/2)
        # layer 4
        src = F.relu(self.layer4(src), inplace=True)  # (N, 256, H/2/2, W/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))  # (N, 256, H/2/2/2, W/2/2)
        # layer 5   batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), inplace=True)  # (N, 512, H/2/2/2, W/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))  # (N, 512, H/2/2/2, W/2/2/2)
        # layer 6   batch norm 3
        src = F.relu(self.batch_norm3(self.layer6(src)), inplace=True)  # (N, 512, H/2/2/2, W/2/2/2)

        all_outputs = []
        for row in range(src.size(2)):  # (N, 512, H/2/2/2, W/2/2/2)
            inp = src[:, :, row, :].permute(2, 0, 1)  # (W/2/2/2, N, 512)
            row_vec = torch.Tensor(batch_size).type_as(inp.data).long().fill_(row)  # type_as主要解决cpu和gpu问题
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat((pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout


if __name__ == '__main__':
    encoder = ImageEncoder(2, True, 512, 0.1, 1)
    print(encoder)
    generator = nn.Sequential(
        nn.Linear(512, 100),
        nn.LogSoftmax(dim=-1)
    )
    encoder.generator = generator
    print(encoder.generator)
