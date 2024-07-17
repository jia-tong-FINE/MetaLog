from CONSTANTS import *
from random import random


def cut_by_613(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.6 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_all(instances):
    np.random.shuffle(instances)
    return instances, [], []


def cut_by_316(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.3 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_by_415(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.4 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_by_514(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.5 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_by_217(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.2 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_by_316_filter(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.3 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_415_filter(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.4 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_226_filter(instances):
    dev_split = int(0.2 * len(instances))
    train_split = int(0.2 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_514_filter(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.5 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_613_filter(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.6 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_172_filter(instances):
    dev_split = int(0.7 * len(instances))
    train_split = int(0.1 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test


def cut_by_253_filter(instances):
    dev_split = int(0.5 * len(instances))
    train_split = int(0.2 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    # train
    temp = []
    for ins in train:
        if ins.label == 'Anomalous':
            ran = random()
            if ran > 0.01:
                continue
        temp.append(ins)
    train = temp
    return train, dev, test