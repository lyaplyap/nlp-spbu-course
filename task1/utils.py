import torch
from torch.nn import functional as F
import sys

def build_dataset(words, stoi, block_size=3):
    X, Y = [], []

    for word in words:
        context = [0] * block_size
        
        for char in word + '.':
            index = stoi[char]
            
            X.append(context)
            Y.append(index)
            
            context = context[1:] + [index]

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    return X_tensor, Y_tensor

def check_loss(parameters, dataset):
    [C, W1, b1, W2, b2] = parameters
    X, Y = dataset

    embed = C[X]
    h = torch.tanh(embed.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2

    return F.cross_entropy(logits, Y).item()

def get_name():
    args = sys.argv
    name = args[1]

    if name in ['-l', '--last', '--lastnames', '--surnames']:
        return 'lastnames'

    if name in ['-f', '--first', '--firstnames', '--names']:
        return 'firstnames'

    raise ValueError('An unknown variable was passed')