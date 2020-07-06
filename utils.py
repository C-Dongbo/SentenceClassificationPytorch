import os
import random


class Data(object):
  def __init__(self, dataPath):
    data = os.path.join(dataPath, 'train_data')
    with open(data, 'rt', encoding='utf-8') as f:
      self.train_data, self.valid_data = get_train_valid_data(f.readlines())






def get_train_valid_data(data :list, ratio=0.9):
  random.shuffle(data)

  train_data =