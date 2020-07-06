# -*- coding: utf-8 -*-


import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

from transformer.config import Config
from dataset import SentenceClassificationDataset, preprocess, collate_fn
from models import Classification, RCNN, Transformer, DualRCNN
from utils import topk_accuracy
from Transformer import Model

def save(dirname,model_name, *args):
    torch.save(model.state_dict(), os.path.join(dirname, 'model_{}.pt'.format(model_name)))

def infer(raw_data, label, **kwargs):
    """
    :param raw_data: raw input (여기서는 문자열)을 입력받습니다
    :param kwargs:
    :return:
    """


    preprocessed_data = preprocess('../data/processing_data',raw_data, config.strmaxlen)
    model.load_state_dict(torch.load("../model/model_RCNN.pt", map_location='cpu'))
    #model.load_state_dict(torch.load("../model/model_RCNN.pt"))

    model.eval()

    output_prediction = model(preprocessed_data)
    point = output_prediction.data.squeeze(dim=1).tolist()

    numpy_point = np.array(point)
    tot = len(point)
    k = 10
    for i in range(1,k+1):
        topk_list = (-numpy_point).argsort()[:,:i].tolist()
        n_correct = 0
        for p, topk in zip(label, topk_list):
            # print('topk prediction = {}, true_label = {}'.format(topk, p))
            if int(p.replace('\n','')) in topk: n_correct += 1
        score = n_correct / tot
        print("accuracy@top{} : {} ".format(i, score))

    point = [np.argmax(p) for p in point]
    
    return list(zip(np.zeros(len(point)), point))




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--output_size', type=int, default=573) #output label size
    args.add_argument('--max_epoch', type=int, default=5)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=300)
    args.add_argument('--model_name', type=str, default='RCNN')


    config = args.parse_args()
    config2 = Config()

    dataset = SentenceClassificationDataset('../data/processing_data', config.strmaxlen)

    print('unique labels = {}'.format(dataset.get_unique_labels_num()))
    print('vocab size = {}'.format(dataset.get_vocab_size()))

    if config.model_name == 'CNN':
        model = Classification(config.embedding, config.strmaxlen, dataset.get_unique_labels_num(),
                               dataset.get_vocab_size())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    elif config.model_name == 'RCNN':
        model = RCNN(config.embedding, config.strmaxlen, dataset.get_unique_labels_num(),
                               dataset.get_vocab_size())
        if config.mode == 'train':
          model = model.cuda()
        
                              
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    elif config.model_name =="DUALRCNN":
        model = DualRCNN(config.embedding, config.strmaxlen, dataset.get_unique_labels_num(),
                               dataset.get_vocab_size())
        if torch.cuda.is_available():
            model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    elif config.model_name == 'TRANSFORMER':
        model = Model(config2)
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)


    if config.mode == 'train':

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=0)
        total_batch = len(train_loader)
        pre_loss = 100000
        for epoch in range(config.max_epoch):
            avg_loss = []
            predictions = []
            label_vars = []
            for i, (data, labels) in enumerate(train_loader):
                # print('i = {}'.format(i))
                # print(len(data))
                predictions = model(data)
                # print(len(predictions))
                loss = criterion(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i%100==0:
                    print('Batch : ', i + 1, '/', total_batch,
                        ', Loss in this minibatch: ', loss.item())
                avg_loss.append(loss.item())
            avg_loss_val = np.array(avg_loss).mean()
            print('epoch:', epoch, ' train_loss:', avg_loss_val)
            if avg_loss_val < pre_loss:
                save("../model/", config.model_name)
                pre_loss = avg_loss_val
            else:
                pre_loss = avg_loss_val


    elif config.mode == 'test':
        with open(os.path.join('../data/processing_data/test', 'test_label'), 'rt', encoding='utf-8') as f2:
            label = f2.readlines()

        with open(os.path.join('../data/processing_data/test', 'test_data'), 'rt', encoding='utf-8') as f:
            dataset = f.readlines()
        res = infer(dataset , label)

        f.close()
        f2.close()

        cnt = 0

        for idx ,result in enumerate(res):
            # print('predict = {}, true = {}'.format(result[1], label[idx]))
            # print(result)
            if result[1] == int(label[idx].replace("\n","")):
                cnt += 1
        
        print('Accuracy : ' + str(cnt/len(label)))


