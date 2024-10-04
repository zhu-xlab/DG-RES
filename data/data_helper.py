from os.path import join

import random
from PIL import Image, ImageOps, ImageFilter

import torch
from torchvision import transforms

from data.concat_dataset import ConcatDataset
from data.single_dataset import SingleDataset
from data.data_split import get_split_domain_info_from_dir, get_split_dataset_info_from_txt, _dataset_info

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
officehome_datasets = ['Art', 'Clipart', 'Product', 'RealWorld']
available_datasets = officehome_datasets + pacs_datasets + vlcs_datasets



def get_train_dataloader(args):
    dataset_path = args.data_root
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    train_datasets = []
    val_datasets = []
    img_transformer_train = get_train_transformers(args)
    img_transformer_val = get_val_transformer(args)

    for i, dname in enumerate(dataset_list):
        name_train, name_val, classes_train, classes_val, domains_train, domains_val = \
            get_split_domain_info_from_dir(join(dataset_path, dname), dataset_name=args.data, \
                                           val_percentage=args.val_perc, domain_label=i+1)
        train_dataset = SingleDataset(name_train, classes_train, domains_train, \
                                     dataset_path=dataset_path, img_transformer=img_transformer_train)
        val_dataset = SingleDataset(name_val, classes_val, domains_val, \
                                    dataset_path=dataset_path, img_transformer=img_transformer_val)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               num_workers=8,
                                               pin_memory=True, 
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             num_workers=8,
                                             pin_memory=True, 
                                             drop_last=False)
    return train_loader, val_loader


def get_target_dataloader(args, shuffle=True):
    dataset_path = args.data_root

    _, names, _, labels, _, domain_labels = get_split_domain_info_from_dir(
        join(dataset_path, args.target), dataset_name=args.data, \
        val_percentage=1.0, domain_label=args.n_domains+1)

    target_img_tr = get_val_transformer(args)
    target_dataset = SingleDataset(names, labels, domain_labels, \
                                   dataset_path=dataset_path, img_transformer=target_img_tr)
    target_dataset = ConcatDataset([target_dataset])
    target_loader = torch.utils.data.DataLoader(target_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=8,
                                                pin_memory=True, 
                                                drop_last=False)
    return target_loader


def get_train_transformers(args):
    img_tr = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale)),
        transforms.RandomHorizontalFlip(args.random_horiz_flip),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter, args.jitter),
        transforms.RandomGrayscale(args.tile_random_grayscale),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return img_tr

def get_val_transformer(args):
    img_tr = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_tr

