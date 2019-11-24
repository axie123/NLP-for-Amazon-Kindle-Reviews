# This is the RNN used for our project:

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.h_dim = hidden_dim
        self.GRU = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths=lengths)
        _, x = self.GRU(embedded)
        x = self.fc1(x)
        return x
