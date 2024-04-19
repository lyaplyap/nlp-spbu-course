import torch
from torch.nn import functional as F
from tqdm import tqdm
import random
import os

from utils import build_dataset, check_loss, get_name

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, 'data')
DIST_PATH = os.path.join(DIRNAME, 'dist')

ITERS_COUNT = 30000
SEED = 1234

def build_vocabulary(words):
    chars = sorted(list(set(''.join(words))))

    stoi = { s: i + 1 for i, s in enumerate(chars) }
    itos = { i: s for s, i in stoi.items() }

    stoi['.'] = 0
    itos[0] = '.'

    return stoi, itos

def build_parameters(stoi_length):
    generator = torch.Generator().manual_seed(SEED)

    C = torch.randn((stoi_length, 10), generator=generator)
    W1 = torch.randn((30, 50), generator=generator)
    b1 = torch.randn(50, generator=generator)
    W2 = torch.randn((50, stoi_length), generator=generator)
    b2 = torch.randn(stoi_length, generator=generator)

    return [C, W1, b1, W2, b2]

def train(parameters, dataset, iters_count):
    for p in parameters:
        p.requires_grad = True

    [C, W1, b1, W2, b2] = parameters
    X_train, Y_train = dataset

    for i in tqdm(range(iters_count)):
        # minibatch construct
        ix = torch.randint(0, X_train.shape[0], (32,))

        # forward pass
        emb = C[X_train[ix]]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y_train[ix])

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    return parameters

def save_vocabulary(stoi, itos, dist_path):
    stoi_path = os.path.join(dist_path, 'stoi.pth')
    itos_path = os.path.join(dist_path, 'itos.pth')
    
    torch.save(stoi, stoi_path)
    print(f'Vocabulary (symbol to index) is saved on the path "{stoi_path}"')

    torch.save(itos, itos_path)
    print(f'Vocabulary (index to symbol) is saved on the path "{itos_path}"')

def save_model(parameters, dist_path):
    model_path = os.path.join(dist_path, 'model.pth')

    [C, W1, b1, W2, b2] = parameters

    model = {
        'C': C,
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    torch.save(model, model_path)
    print(f'Trained model is saved on the path "{model_path}"')

def main():
    name = get_name()

    data_file_path = os.path.join(DATA_PATH, name, 'train.txt')
    dist_dir_path = os.path.join(DIST_PATH, name)

    words = open(data_file_path, 'r').read().splitlines()

    random.seed(SEED)
    random.shuffle(words)

    stoi, itos = build_vocabulary(words)
    save_vocabulary(stoi, itos, dist_dir_path)

    parameters = build_parameters(len(stoi))
    dataset = build_dataset(words, stoi)

    train(parameters, dataset, ITERS_COUNT)
    print(f'Model is trained, loss: {check_loss(parameters, dataset):.2f}')

    save_model(parameters, dist_dir_path)

main()
