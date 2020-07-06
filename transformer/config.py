# config.py
import torch

class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout = 0.5
    require_improvement = 2000
    num_classes = 508
    n_vocab = 1690
    num_epochs = 20
    batch_size = 128
    pad_size = 200
    learning_rate = 5e-4
    embed = 300
    dim_model = 300
    hidden = 1024
    last_hidden = 512
    num_head = 5
    num_encoder = 2