import torch
import torch.optim as optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchtext
from torchtext import data
import spacy

import argparse
import os

from models import *

torch.manual_seed(0)


def evaluate(model, loader, loss_fnc):
    total_corr = 0
    running_loss = 0
    for i, vbatch in enumerate(loader):
        input, input_length = vbatch.text
        label = vbatch.label
        prediction = model(input, input_length)
        loss = loss_fnc(input=prediction.squeeze(), target=label.float())
        running_loss += loss
        corr = (prediction > 0.5).squeeze().long() == label.long()
        total_corr += int(corr.sum())
    return (float(total_corr) / len(loader.dataset)), (running_loss / len(loader.dataset))


def main(args):
    ######
    # <editor-fold desc="Description">
    # 3.2.1
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True,
                      tokenizer_language='en_core_web_sm')
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='train.csv',
        validation='validation.csv', test='test.csv', format='csv', skip_header=True,
        fields=[('text', TEXT), ('label', LABELS)])

    # To create overfit
    '''
    overfit_data, val_data2, test_data2 = data.TabularDataset.splits(
        path='data/', train='overfit.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv', skip_header=True,
        fields=[('text', TEXT), ('label', LABELS)])
    '''
    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    '''
    t_iter = data.Iterator(dataset = train_data, batch_size= args.batch_size,
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # To create overfit

    overfit_iter, val_iter2, test_iter2 = data.BucketIterator.splits(
        (overfit_data, val_data2, test_data2), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    '''
    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # For overfit
    # TEXT.build_vocab(overfit_data, val_data2, test_data2)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:", TEXT.vocab.vectors.shape)
    # </editor-fold>
    ######

    # 5 Training and Evaluation

    ######
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    # model = args.model
    model = 'cnn'
    overfit = 0
    itera = 0
    if model == 'baseline':
        net = Baseline(embdim, vocab)
    elif model == 'cnn':
        net = CNN(embdim, vocab, 50, [2, 4])
    elif model == 'rnn':
        net = RNN(embdim, vocab, hiddendim)

    loss_fnc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    t = 0
    vstep = []
    vacc = []
    vloss = []

    tstep = []
    tacc = []
    tloss = []

    testep = []
    teacc = []
    teloss = []

    j = 0
    times = []

    evalevery = 2

    # Training Loop
    if overfit == 0:
        if model == 'baseline':
            print("Baseline Loop")

        if model == 'cnn':
            print("CNN Loop")

        if model == 'rnn':
            print("RNN Loop")
        for epoch in range(epochs):

            running_loss = 0
            total_corr = 0
            for i, d in enumerate(train_iter):
                b_input, b_input_length = d.text
                label = d.label
                optimizer.zero_grad()
                predictions = net(b_input, b_input_length)
                loss = loss_fnc(input=predictions.squeeze(), target=label.float())
                predictions = torch.sigmoid(predictions)
                running_loss += loss

                loss.backward()
                optimizer.step()
                corr = (predictions > 0.5).squeeze().long() == label.long()
                total_corr += int(corr.sum())

                if (epoch) % evalevery == 0:
                    valid_Acc, valid_loss = evaluate(net, val_iter, loss_fnc)
                    vstep.append(epoch + 1)
                    vacc.append(valid_Acc)
                    vloss.append(valid_loss)

                    train_Acc, train_loss = evaluate(net, train_iter, loss_fnc)
                    tstep.append(epoch + 1)
                    tacc.append(train_Acc)
                    tloss.append(train_loss)

                    test_Acc, test_loss = evaluate(net, test_iter, loss_fnc)
                    testep.append(epoch + 1)
                    teacc.append(test_Acc)
                    teloss.append(test_loss)

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Train acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              train_Acc))
                    print(
                        "Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              valid_Acc))

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Test acc: {}".format(epoch + 1, t + 1,
                                                                             (running_loss / evalevery),
                                                                             test_Acc))
                    running_loss = 0

                    if valid_Acc >= 0.98:
                        j += 1
                        if j > 10:
                            break

                t = t + 1

    # Loop for overfit part
    if overfit == 1:
        if model == 'baseline':
            print("Baseline Loop (overfit)")

        if model == 'cnn':
            print("CNN Loop (overfit)")

        if model == 'rnn':
            print("RNN Loop (overfit)")
        for epoch in range(epochs):

            running_loss = 0
            total_corr = 0
            for i, d in enumerate(overfit_iter):
                b_input, b_input_length = d.text
                label = d.label
                optimizer.zero_grad()
                predictions = net(b_input, b_input_length)
                loss = loss_fnc(input=predictions.squeeze(), target=label.float())
                predictions = torch.sigmoid(predictions)
                running_loss += loss

                loss.backward()
                optimizer.step()
                corr = (predictions > 0.5).squeeze().long() == label.long()
                total_corr += int(corr.sum())

                if (epoch) % evalevery == 0:
                    valid_Acc, valid_loss = evaluate(net, val_iter2, loss_fnc)
                    vstep.append(epoch + 1)
                    vacc.append(valid_Acc)
                    vloss.append(valid_loss)

                    train_Acc, train_loss = evaluate(net, overfit_iter, loss_fnc)
                    tstep.append(epoch + 1)
                    tacc.append(train_Acc)
                    tloss.append(train_loss)

                    test_Acc, test_loss = evaluate(net, test_iter2, loss_fnc)
                    testep.append(epoch + 1)
                    teacc.append(test_Acc)
                    teloss.append(test_loss)

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Train acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              train_Acc))
                    print(
                        "Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              valid_Acc))

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Test acc: {}".format(epoch + 1, t + 1,
                                                                             (running_loss / evalevery),
                                                                             test_Acc))
                    running_loss = 0

                    if valid_Acc >= 0.98:
                        j += 1
                        if j > 10:
                            break

                t = t + 1

    # Loop for iterator
    if itera == 1:
        if model == 'baseline':
            print("Baseline Loop (Iterator)")

        if model == 'cnn':
            print("CNN Loop (Iterator)")

        if model == 'rnn':
            print("RNN Loop (Iterator)")
        for epoch in range(epochs):

            running_loss = 0
            total_corr = 0
            for i, d in enumerate(t_iter):
                b_input, b_input_length = d.text
                label = d.label
                optimizer.zero_grad()
                predictions = net(b_input, b_input_length)
                loss = loss_fnc(input=predictions.squeeze(), target=label.float())
                predictions = torch.sigmoid(predictions)
                running_loss += loss

                loss.backward()
                optimizer.step()
                corr = (predictions > 0.5).squeeze().long() == label.long()
                total_corr += int(corr.sum())

                if (epoch) % evalevery == 0:
                    valid_Acc, valid_loss = evaluate(net, val_iter, loss_fnc)
                    vstep.append(epoch + 1)
                    vacc.append(valid_Acc)
                    vloss.append(valid_loss)

                    train_Acc, train_loss = evaluate(net, t_iter, loss_fnc)
                    tstep.append(epoch + 1)
                    tacc.append(train_Acc)
                    tloss.append(train_loss)

                    test_Acc, test_loss = evaluate(net, test_iter, loss_fnc)
                    testep.append(epoch + 1)
                    teacc.append(test_Acc)
                    teloss.append(test_loss)

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Train acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              train_Acc))
                    print(
                        "Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch + 1, t + 1,
                                                                              (running_loss / evalevery),
                                                                              valid_Acc))

                    print(
                        "Epoch: {}, Step {} | Loss: {}| Test acc: {}".format(epoch + 1, t + 1,
                                                                             (running_loss / evalevery),
                                                                             test_Acc))
                    running_loss = 0

                    if valid_Acc >= 0.98:
                        j += 1
                        if j > 10:
                            break

                t = t + 1
    # Return Loss and Accuracy
    print("Training Loss: " + str(tloss[-1].detach().numpy()) + " " +
          "Validation Loss: " + str(vloss[-1].detach().numpy()) + " " +
          "Test Loss: " + str(teloss[-1].detach().numpy()))

    print("Training Accuracy: " + str(tacc[-1]) + " " +
          "Validation Accuracy: " + str(vacc[-1]) + " " +
          "Test Accuracy: " + str(teacc[-1]))

    # Print plots
    # <editor-fold desc="Description">
    plt.plot(tstep, tloss, label='Train')
    plt.plot(tstep, vloss, label='Validation')
    # plt.plot(testep, tloss, label = "Test")
    plt.title('Loss vs. Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(tstep, tacc, label='Train')
    plt.plot(tstep, vacc, label='Validation')
    # plt.plot(tstep, teacc, label='Test')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    if overfit != 1:
        if model == "baseline" and overfit == 0 and itera != 1:
            torch.save(net, 'model_baseline.pt')
        elif model == "cnn" and overfit == 0 and itera != 1:
            print("Saving CNN model")
            torch.save(net, 'model_cnn.pt')
        elif model == 'rnn' and overfit == 0 and itera != 1:
            print("Saving RNN model")
            torch.save(net, 'model_rnn.pt')
    # </editor-fold>


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='cnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()
    embdim = 100
    hiddendim = 100

    main(args)
