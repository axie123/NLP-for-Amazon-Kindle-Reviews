import torchtext
from torchtext import data
import spacy

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
import matplotlib.pyplot as plt

from model_RNN import RNN

def main(args):

    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
                path='data/', train='train.csv',
                validation='validation.csv', test='test.csv', format='csv',
                skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    TEXT.build_vocab(train_data, val_data, test_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:", TEXT.vocab.vectors.shape)

    # Model:
    model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    def evaluate(model, val_loader):
        v_accuracy = 0
        vloss = 0
        batch_t = 0
        for index, batch in enumerate(val_loader):
            data, length = batch.text
            truth = batch.label.float()
            y_valid = model(data, length).float()
            v_loss = bce_loss(y_valid.squeeze(), truth)
            vloss += v_loss.item()
            pred = (y_valid > 0.5).squeeze().float() == truth
            v_accuracy += int(pred.sum())
            batch_t += 1

        return v_accuracy / len(val_loader.dataset), vloss / batch_t

    # Training loop used for all models:
    training_accuracy = []
    training_loss = []
    valid_acc = []
    valid_loss = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch_tracker = 0
        for index, batch in enumerate(train_iter):
            batch_input, batch_input_length = batch.text
            labels = batch.label.float()
            optimizer.zero_grad()
            y = model(batch_input, batch_input_length).float()
            loss = bce_loss(input=y.squeeze(), target=labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = (y > 0.5).squeeze().float() == labels
            epoch_accuracy += int(pred.sum()) / len(labels)
            batch_tracker += 1
        training_loss += [epoch_loss / batch_tracker]
        training_accuracy += [epoch_accuracy / batch_tracker]
        # Evaluating the validation and testing data, as well as giving the losses and accuracies for all 3:
        if (epoch + 1) % 1 == 0:
            print("Epoch: ", f'{epoch + 1}', ", Training Loss: ", f'{epoch_loss / batch_tracker:.4f}',
                  ", Training Accuracy: ", f'{epoch_accuracy / batch_tracker:.4f}')
            valid_accuracy, v_loss = evaluate(model, val_iter)
            valid_acc += [valid_accuracy]
            valid_loss += [v_loss]
            print("Epoch: ", f'{epoch + 1}', ", Validation Loss: ", f'{v_loss:.4f}',
                  ", Validation Accuracy: ", f'{valid_accuracy:.4f}')

    # Plotting the accuracies and loss of training, validation, and test datasets:
    epoch_idx = np.arange(0, len(training_accuracy))

    # Training-Validation Accuracy
    plt.plot(epoch_idx, training_accuracy)
    plt.plot(epoch_idx, valid_acc)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title("RNN Training and Validation Accuracy on Overfitting")
    plt.show()

    # Training-Validation Loss
    plt.plot(epoch_idx, training_loss)
    plt.plot(epoch_idx, valid_loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title("RNN Training and Validation Loss on Overfitting")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('epochs', type=int, default=25)
    parser.add_argument('model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
