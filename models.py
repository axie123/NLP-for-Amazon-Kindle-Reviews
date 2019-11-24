import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # x = [sentence length, batch size]
        embedded = self.embedding(x)
        average = embedded.mean(0)  
        output = self.fc(average).squeeze(1)
        return output


class CNN(nn.Module):
    
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[0], embedding_dim), stride=1)
        self.conv2 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[1], embedding_dim), stride=1)
        self.fc1 = nn.Linear(len(filter_sizes) * 50, 1)

    def forward(self, x, lengths=None):
        x = torch.transpose(x, 0, 1)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        h1 = F.relu((self.conv1(x)).squeeze(3))
        h1 = F.max_pool1d(h1, h1.size()[2]).squeeze(2)
        h2 = F.relu(self.conv2(x).squeeze(3))
        h2 = F.max_pool1d(h2, h2.size()[2]).squeeze(2)
        x = torch.cat((h1, h2), 1)
        x = self.fc1(x).squeeze(1)
        return x


class RNN(nn.Module):
    
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.n_hidden = hidden_dim
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths=None):
        bs = x.size(1)
        self.h = self.init_hidden(bs)
        embs = self.embedding(x)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        output = self.fc1(self.h[-1]).squeeze(1)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.n_hidden)
