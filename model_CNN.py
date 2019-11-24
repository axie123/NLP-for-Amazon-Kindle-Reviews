# This file contains the detailed CNN used for the project.

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(filter_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, kernel_size=(filter_size[1], embedding_dim))
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        embedded = torch.reshape(embedded, (embedded.shape[0], 1, embedded.shape[1], embedded.shape[2]))
        x1 = F.relu(self.conv1(embedded))
        max_pool1 = nn.MaxPool2d((x1.shape[2], 1), stride=1)
        x1 = max_pool1(x1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1]))
        x2 = F.relu(self.conv2(embedded))
        max_pool2 = nn.MaxPool2d((x2.shape[2], 1), stride=1)
        x2 = max_pool2(x2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1]))
        y = torch.cat((x1, x2), dim=1)
        y_final = self.fc1(y)
        return y_final