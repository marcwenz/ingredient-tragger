import torch
import torch.nn as nn

# TODO Implement model from 'scratch'


class CRF(nn.Module):
    """
    Needed:
        ** viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
        ** forward algorithm: https://github.com/kaniblu/pytorch-bilstmcrf/blob/6573a6efc54f9cea06e8be346d01fcce1d8e393f/model.py#L27
        ** transition score
    """
    pass


class BiLSTMCRF(nn.Module):
    """
    Needed:
        ** init w/ layer definition
        ** embeddings
        ** forward algorithm
        ** bilstm score
        ** connection to CRF?
    """
    pass
