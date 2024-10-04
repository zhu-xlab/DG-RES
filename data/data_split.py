import sys
import os
import numpy as np
from random import sample, random


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    # read from the official split txt
    file_names = []
    labels = []

    for row in open(txt_labels, 'r'):
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def find_classes(dir_name):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
    classes.sort()
    class_to_idx = {classes[i]: i+1 for i in range(len(classes))}
    return classes, class_to_idx


def get_split_domain_info_from_dir(domain_path, dataset_name=None, val_percentage=None, domain_label=None):
    # read from the directory
    domain_name = domain_path.split("/")[-1]

    names, labels = [], []
    classes, class_to_idx = find_classes(domain_path)
    for i, item in enumerate(classes):
        class_path = domain_path + "/" + item
        for root, _, fnames in sorted(os.walk(class_path)):
            for fname in sorted(fnames):
                path = os.path.join(domain_name, item, fname)
                names.append(path)
                labels.append(class_to_idx[item])

    name_train, name_val, labels_train, labels_val = get_random_subset(names, labels, val_percentage)
    domain_label_train = [domain_label for i in range(len(labels_train))]
    domain_label_val = [domain_label for i in range(len(labels_val))]
    return name_train, name_val, labels_train, labels_val, \
           domain_label_train, domain_label_val

def get_split_dataset_info_from_txt(txt_path, domain, domain_label, val_percentage=None):
    if "PACS" in txt_path:
        train_name = "_train_kfold.txt"
        val_name = "_crossval_kfold.txt"

        train_txt = txt_path + "/" + domain + train_name
        val_txt = txt_path + "/" + domain + val_name

        train_names, train_labels = _dataset_info(train_txt)
        val_names, val_labels = _dataset_info(val_txt)
        train_domain_labels = [domain_label for i in range(len(train_labels))]
        val_domain_labels = [domain_label for i in range(len(val_labels))]
        return train_names, val_names, train_labels, val_labels, \
               train_domain_labels, val_domain_labels

    elif "miniDomainNet" in txt_path:
        # begin at 0, need to add 1
        train_name = "_train.txt"
        val_name = "_test.txt"
        train_txt = txt_path + "/" + domain + train_name
        val_txt = txt_path + "/" + domain + val_name

        train_names, train_labels = _dataset_info(train_txt)
        val_names, val_labels = _dataset_info(val_txt)
        train_labels = [label + 1 for label in train_labels]
        val_labels = [label + 1 for label in val_labels]

        names = train_names + val_names
        labels = train_labels + val_labels
        train_names, val_names, train_labels, val_labels = get_random_subset(names, labels, val_percentage)

        train_domain_labels = [domain_label for i in range(len(train_labels))]
        val_domain_labels = [domain_label for i in range(len(val_labels))]
        return train_names, val_names, train_labels, val_labels, \
               train_domain_labels, val_domain_labels
    else:
        raise NotImplementedError


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)
