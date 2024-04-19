import torch
from torch.nn import functional as F
import os

from utils import build_dataset, check_loss, get_name

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, 'data')
DIST_PATH = os.path.join(DIRNAME, 'dist')

ITERS_COUNT = 30000
SEED = 1234

def prettify(name):
    return name[0].upper() + name[1: -1]

def generate_names(parameters, itos, count=5, block_size=3):
    generator = torch.Generator().manual_seed(SEED)

    [C, W1, b1, W2, b2] = parameters
    names = []

    for _ in range(count):
        out = []
        context = [0] * block_size
        
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1, generator=generator).item()
            context = context[1:] + [index]
            out.append(index)
            
            if index == 0:
                name = prettify(''.join(itos[i] for i in out))
                names.append(name)

                break
    
    return names

def main():
    name = get_name()

    data_file_path = os.path.join(DATA_PATH, name, 'test.txt')
    model_path = os.path.join(DIST_PATH, name, 'model.pth')
    stoi_path = os.path.join(DIST_PATH, name, 'stoi.pth')
    itos_path = os.path.join(DIST_PATH, name, 'itos.pth')

    words = open(data_file_path, 'r').read().splitlines()
    parameters = torch.load(model_path)
    stoi = torch.load(stoi_path)
    itos = torch.load(itos_path)

    C = parameters['C']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    parameters = [C, W1, b1, W2, b2]
    dataset = build_dataset(words, stoi)
    
    print(f'Test loss: {check_loss(parameters, dataset):.2f}\n')
    
    print('Generated names:')
    for name in generate_names(parameters, itos):
        print(f'* {name}')

main()
