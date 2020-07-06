# -*- coding: utf-8 -*-

import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch.autograd import Variable



class Classification(nn.Module):
    def __init__(self, embedding_dim, max_length, output_size, vocab_size):
        """
        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        :param output_size: 출력 라벨의 갯수입니다
        """
        super(Classification, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = vocab_size + 3
        self.output_dim = output_size  # number of labels
        self.max_length = max_length

        windows = [3, 4, 5]
        num_filters = 128

        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)
        self.convs = [nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(n, embedding_dim), bias=False) for
                      n in windows]
        for idx, conv in enumerate(self.convs):
            self.add_module('conv%d' % idx, conv)
        self.fc = nn.Linear(len(windows) * num_filters, self.output_dim, bias=False)
        self.ranges = np.asarray([n for n in windows for _ in range(num_filters)])

    def forward(self, data: list):
        """
        :param data: 실제 입력값
        :return:
        """
        batch_size = len(data)
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())

        embeds = self.embeddings(data_in_torch)
        embeds = torch.reshape(embeds, shape=(batch_size, 1, self.max_length, self.embedding_dim))
        out = [torch.nn.functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        out = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        output = self.fc(out)
        return output


class RCNN(nn.Module):
    def __init__(self, embedding_dim, max_length, output_size, vocab_size):
        super(RCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.character_size = vocab_size + 3
        self.output_dim = output_size  # number of labels
        self.max_length = max_length

        windows = [3, 4, 5]
        num_filters = 128

        # Embedding Layer
        self.embeddings = nn.Embedding(self.character_size, embedding_dim)

        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=150,
                            num_layers=1,
                            dropout=0.7,
                            bidirectional=True)

        self.dropout = nn.Dropout(0.7)


        self.convs = [nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(n, embedding_dim), bias=False) for
                      n in windows]
        for idx, conv in enumerate(self.convs):
            self.add_module('conv%d' % idx, conv)
        self.fc = nn.Linear(len(windows) * num_filters, self.output_dim, bias=False)
        self.ranges = np.asarray([n for n in windows for _ in range(num_filters)])

    def forward(self, x):
        print(x)
        batch_size = x.size(0)
        # batch_size =len(x)
        # x.shape = (seq_len, batch_size)
        embeds = self.embeddings(x)

        # embeds.shape = (batch_size, seq_len, embed_size)
        embeds, (h_n, c_n) = self.lstm(embeds)

        # embeds.shape = (seq_len, batch_size, 2 * hidden_size)

        embeds = torch.reshape(embeds, shape=(batch_size, 1, self.max_length, self.embedding_dim))

        out = [torch.nn.functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        out = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        output = self.fc(out)
        return output


class DualRCNN(nn.Module):
    def __init__(self, embedding_dim, max_length, output_size, vocab_size):
        super(DualRCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.character_size = vocab_size + 3
        self.output_dim = output_size  # number of labels
        self.max_length = max_length

        windows = [3, 4, 5]
        num_filters = 128

        # Embedding Layer
        self.embeddings = nn.Embedding(self.character_size, embedding_dim)

        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=150,
                            num_layers=1,
                            dropout=0.7,
                            bidirectional=True)



        self.dropout = nn.Dropout(0.7)

        self.convs = [nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(n, embedding_dim), bias=False) for
                      n in windows]
        for idx, conv in enumerate(self.convs):
            self.add_module('conv%d' % idx, conv)
        self.fc = nn.Linear(len(windows) * num_filters, self.output_dim, bias=False)
        self.ranges = np.asarray([n for n in windows for _ in range(num_filters)])

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # x.shape = (seq_len, batch_size)
        embeds1 = self.embeddings(x1)
        embeds2 = self.embeddings(x2)

        # embeds.shape = (batch_size, seq_len, embed_size)
        embeds1, (h_n, c_n) = self.lstm(embeds1)
        embeds2, (h_n, c_n) = self.lstm(embeds2)

        print("embeds1 shape : {}, embeds2 shape : {}".format(embeds1.shape, embeds2.shape))

        embeds = torch.cat((embeds1, embeds2),dim=2)
        print(embeds.shape)
        # embeds.shape = (seq_len, batch_size, 2 * hidden_size)

        embeds = torch.reshape(embeds, shape=(batch_size, 1, self.max_length, embeds.shape[2]))

        out = [torch.nn.functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        out = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        output = self.fc(out)
        return output
