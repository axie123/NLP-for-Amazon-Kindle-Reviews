import torch.nn as nn

class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        average = embedded.mean(0)
        output = self.fc(average).squeeze(1)
        return output