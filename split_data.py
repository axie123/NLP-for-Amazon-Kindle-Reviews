"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""
import pandas as pd
import gzip
import numpy as np


def parse(path):
    g = gzip.open(path, 'rb')
    j = 0
    for l in g:
        yield eval(l)
        j = j + 1
        if j > 100000:
            break


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def createData(path):
    df = getDF(path);

    """
    df2 = df.copy()
    df2.loc[df2['overall'] < 3, 'overall'] = -1
    df2.loc[df2['overall'] == 3, 'overall'] = 0
    df2.loc[df2['overall'] > 3, 'overall'] = 1
    df2['overall'] = pd.Categorical(df2['overall'])
    dfd = pd.get_dummies(df2['overall'], prefix='pref')
    dff = pd.concat([df2, dfd], axis=1)
    dff.rename(columns={"pref_-1.0": "negative", "pref_0.0": "neutral", "pref_1.0": "positive"}, inplace=True)
    """

    df = df[["reviewText", "overall"]]
    df["overall"] = round(df["overall"] / 5, 0)
    df = df[df['reviewText'].map(len) <= 2000]
    df = df[df['reviewText'].map(len) >= 10]

    df = df[df['reviewText'].map(len) > 0]

    # Equal sample size

    df['weights'] = (df['overall'] * -100) + 101
    x = df.sample(df['overall'].value_counts()[0] * 2, weights=df['weights']);

    x.rename(columns={"reviewText": "text", "overall": "label"}, inplace=True)
    print(x['label'].value_counts())
    x.to_csv("data/data.csv", index=False)


def createData2(path):
    df = getDF(path);
    df = df[["reviewText", "overall"]]
    df["sentiment"] = round(df["overall"] / 5, 0)
    df = df[df['reviewText'].map(len) <= 2000]

    df = df[df['reviewText'].map(len) > 0]

    df['weights'] = (df['sentiment'] * -100) + 101
    df = df.sample(df['sentiment'].value_counts()[0] * 2, weights=df['weights']);

    df2 = df.copy()
    df2.loc[df2['overall'] < 3, 'overall'] = -1
    df2.loc[df2['overall'] == 3, 'overall'] = 0
    df2.loc[df2['overall'] > 3, 'overall'] = 1
    df2['overall'] = pd.Categorical(df2['overall'])
    dfd = pd.get_dummies(df2['overall'], prefix='pref')
    dff = pd.concat([df2, dfd], axis=1)
    dff.rename(columns={"pref_-1.0": "negative", "pref_0.0": "neutral", "pref_1.0": "positive"}, inplace=True)

    # Equal sample size

    dff.rename(columns={"reviewText": "text", "overall": "label"}, inplace=True)

    dff.to_csv("data/data.csv", index=False)
    dff = dff[['text', 'negative', 'neutral', 'positive']]
    return dff

def split():
    df = pd.read_csv('data/data.csv', header=0)
    df = df[["text", 'label']]
    df = df.sort_values('label')

    n = len(df)
    train_amount = n * 70 // 100
    validation_amount = n * 20 // 100
    test_amount = n * 10 // 100

    trainset = df.iloc[0:train_amount // 2]
    trainset = trainset.append(df[n // 2:n // 2 + train_amount // 2])

    valset = df.iloc[train_amount // 2:train_amount // 2 + validation_amount // 2]
    valset = valset.append(df.iloc[n // 2 + train_amount // 2:n // 2 + train_amount // 2 + validation_amount // 2])

    testset = df.iloc[
              validation_amount // 2 + train_amount // 2:validation_amount // 2 + train_amount // 2 + test_amount // 2]
    testset = testset.append(df.iloc[
                             n // 2 + train_amount // 2 + validation_amount // 2:n // 2 + train_amount // 2 + validation_amount // 2 + test_amount // 2])
    overset = df.iloc[0:50]
    overset = overset.append(df.iloc[n // 2:n // 2 + 50])

    trainset = trainset.astype({'label': int})
    testset = testset.astype({'label': int})
    valset = valset.astype({'label': int})

    trainset.to_csv('data/train.csv', sep=',', index=False)
    valset.to_csv('data/validation.csv', sep=',', index=False)
    testset.to_csv('data/test.csv', sep=',', index=False)
    overset.to_csv('data/overfit.csv', sep=',', index=False)

    print("Train Set: " + str(trainset['label'].value_counts()[0]) + " negative and " + str(
        trainset['label'].value_counts()[1]) + " positive")
    print("Validation Set: " + str(valset['label'].value_counts()[0]) + " negative and " + str(
        valset['label'].value_counts()[1]) + " positive")
    print("Test Set: " + str(testset['label'].value_counts()[0]) + " negative and " + str(
        testset['label'].value_counts()[1]) + " positive")

    return trainset, valset, testset, overset

def split2():
    df = pd.read_csv('data/data.csv', header=0)
    df = df[["text", 'negative', 'neutral', 'positive']]
    df = df.sort_values('negative')

    n = len(df)
    train_amount = n * 70 // 100
    validation_amount = n * 20 // 100
    test_amount = n * 10 // 100

    trainset = df.iloc[0:train_amount // 2]
    trainset = trainset.append(df[n // 2:n // 2 + train_amount // 2])

    valset = df.iloc[train_amount // 2:train_amount // 2 + validation_amount // 2]
    valset = valset.append(df.iloc[n // 2 + train_amount // 2:n // 2 + train_amount // 2 + validation_amount // 2])

    testset = df.iloc[
              validation_amount // 2 + train_amount // 2:validation_amount // 2 + train_amount // 2 + test_amount // 2]
    testset = testset.append(df.iloc[
                             n // 2 + train_amount // 2 + validation_amount // 2:n // 2 + train_amount // 2 + validation_amount // 2 + test_amount // 2])
    overset = df.iloc[0:50]
    overset = overset.append(df.iloc[n // 2:n // 2 + 50])

    trainset = trainset.astype({'negative': int})
    trainset = trainset.astype({'neutral': int})
    trainset = trainset.astype({'positive': int})

    testset = testset.astype({'negative': int})
    testset = testset.astype({'neutral': int})
    testset = testset.astype({'positive': int})

    valset = valset.astype({'negative': int})
    valset = valset.astype({'neutral': int})
    valset = valset.astype({'positive': int})

    trainset.to_csv('data/train.csv', sep=',', index=False)
    valset.to_csv('data/validation.csv', sep=',', index=False)
    testset.to_csv('data/test.csv', sep=',', index=False)
    overset.to_csv('data/overfit.csv', sep=',', index=False)

    return trainset, valset, testset, overset

path = 'data/reviews_Books_5.json.gz'
#createData(path)
createData2(path)
split2()
