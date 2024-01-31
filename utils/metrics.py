import torch
from pypesq import pesq
from pystoi import stoi  # pip install pystoi

def pesq_metric(y_hat, bd, sample_rate):
    with torch.no_grad():
        y_hat = y_hat.cpu().numpy()
        y = bd['y'].cpu().numpy()  # target signal

        sum = 0
        for i in range(len(y)):
            sum += pesq(y[i, 0], y_hat[i, 0], sample_rate)

        sum /= len(y)
        return torch.tensor(sum)

def compute_stoi(y_hat, y, sr):
    return stoi(y, y_hat, sr)
