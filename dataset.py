# -*- coding: utf-8 -*-


import os

import numpy as np
from torch.utils.data import Dataset
import torch


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset_path: str, max_length: int):
        """
        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        data_sentences = os.path.join(dataset_path, 'train', 'new_train')
        data_label = os.path.join(dataset_path, 'train', 'new_label')
        self.dataset_path = dataset_path
        with open(data_sentences, 'rt', encoding='utf-8') as f:
            self.sentences1, self.sentences2 = preprocess(dataset_path, f.readlines(), max_length)

        with open(data_label, encoding='utf-8') as f:
            self.labels = [np.float32(x) for x in f.readlines()]
        print('sentences :', len(self.sentences1))
        print('labels :', len(self.labels))

    def __len__(self):
        """
        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.sentences1)

    def __getitem__(self, idx):
        """
        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.sentences1[idx], self.sentences2[idx], self.labels[idx]

    def get_unique_labels_num(self):
        return len(set(self.labels))

    def get_vocab_size(self):
        vocab_path = os.path.join(self.dataset_path, 'vocab.txt')
        lines = open(vocab_path, 'r', encoding='utf-8').readlines()
        return len(lines)

def preprocess(dataset_path: str, data: list, max_length: int):
    """
    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """

    unk = '<unk>'
    pad = '<pad>'
    space = '<space>'

    vocab = {}
    vocab[pad] = 0
    vocab[unk] = 1
    vocab[space] = 2

    fr = open(os.path.join(dataset_path,'vocab.txt'), 'r', encoding='utf-8')
    lines = fr.readlines()
    fr.close()

    vocab_id = 3
    for line in lines:
        vocab[line.replace('\n', '')] = vocab_id
        vocab_id += 1

    vectorized_data1 = []
    vectorized_data2 = []
    for datum in data:
        sent = datum.replace('\n', '').replace('\r', '')
        sent1, sent2 = sent.split("<sep>")

        vec = []
        for char in sent1.strip():
            if char == ' ':
                char = space
            elif char in vocab.keys():
                vec.append(vocab[char])
            else:
                vec.append(vocab[unk])
        vectorized_data1.append(vec)

        vec = []
        for char in sent2.strip():
            if char == ' ':
                char = space
            elif char in vocab.keys():
                vec.append(vocab[char])
            else:
                vec.append(vocab[unk])
        vectorized_data2.append(vec)

    zero_padding1 = np.zeros((len(data), max_length), dtype=np.int32)
    zero_padding2 = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data1):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding1[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding1[idx, :length] = np.array(seq)
    for idx, seq in enumerate(vectorized_data2):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding2[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding2[idx, :length] = np.array(seq)

    return zero_padding1, zero_padding2



def collate_fn(data: list):
    """
    :param data: 데이터 리스트
    :return:
    """
    sentences1 = []
    sentences2 = []
    label = []
    for datum in data:
        sentences1.append(datum[0])
        sentences2.append(datum[1])
        label.append(datum[2])

    if torch.cuda.is_available():
        sent2tensor1 = torch.tensor(sentences1, device='cuda', dtype = torch.float)
        sent2tensor2 = torch.tensor(sentences2, device='cuda', dtype=torch.float)
        label2tensor = torch.tensor(label, device='cuda', dtype=torch.long)
    else:
        sent2tensor1 = torch.tensor(sentences1, device='cpu', dtype = torch.long)
        sent2tensor2 = torch.tensor(sentences2, device='cpu', dtype=torch.long)
        label2tensor = torch.tensor(label, device='cpu', dtype=torch.long)

    # sent2tensor = torch.tensor(sentences, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)
    # label2tensor = torch.tensor(label, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
    return sent2tensor1, sent2tensor2 ,label2tensor
