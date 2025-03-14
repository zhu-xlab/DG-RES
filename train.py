# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]

    # build train dataloader
    train_envs = torch.utils.data.ConcatDataset(
        [env for i, (env, env_weights) in enumerate(in_splits) \
            if i not in args.test_envs])
    indices = list(range(len(train_envs)))
    random.shuffle(indices)
    train_envs = torch.utils.data.Subset(train_envs, indices)
    train_loaders = torch.utils.data.DataLoader(
        train_envs, 
        batch_size=hparams['batch_size'], 
        num_workers=dataset.N_WORKERS,
        drop_last=False,
        shuffle=True)

    # build eval dataloader
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    # build algorithm
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    algorithm.to(device)

    # get training status
    minibatches_iterator = iter(train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = len(train_loaders)
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    val_acc = 0
    last_results_keys = None
    acc_list = {'val_acc': [], 'test_acc': [], 'similarity': []}
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # train data_iter
        try:    
            x, y = next(minibatches_iterator)
        except StopIteration:
            train_envs = torch.utils.data.ConcatDataset(
                [env for i, (env, env_weights) in enumerate(in_splits) \
                    if i not in args.test_envs])
            indices = list(range(len(train_envs)))
            random.shuffle(indices)
            train_envs = torch.utils.data.Subset(train_envs, indices)
            train_loaders = torch.utils.data.DataLoader(
                train_envs, 
                batch_size=hparams['batch_size'], 
                num_workers=dataset.N_WORKERS,
                drop_last=False,
                shuffle=True
            )
            minibatches_iterator = iter(train_loaders)
            x, y = next(minibatches_iterator)

        minibatches = [(x.to(device), y.to(device))]
        step_vals = algorithm.update(minibatches)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # algorithm.update_ema_hist()

            val_acc_total, val_env_total, test_acc = 0, 0, 0
            feats_val = torch.zeros((0, 2048)).cuda()
            feats_test = torch.zeros((0, 2048)).cuda()
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc, correct_num, total_num = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                if str(args.test_envs[0]) in name:
                    if '_in' in name:
                        test_acc = acc
                else:
                    if '_out' in name:
                        val_acc_total += acc
                        val_env_total += 1.0

            val_acc = val_acc_total / val_env_total
            acc_list['val_acc'].append(val_acc)
            acc_list['test_acc'].append(test_acc)
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    max_val_acc = 0
    max_test_acc = 0
    for idx, val_acc in enumerate(acc_list['val_acc']):
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_test_acc = acc_list['test_acc'][idx]
    print ('max_val_acc: ', max_val_acc, ' max_test_acc: ', max_test_acc)

    with open('{}_env_{}_{}_val_acc.txt'.format(args.dataset, \
                str(args.test_envs[0]), args.algorithm), 'w') as file:
        for number in acc_list['val_acc']:
            file.write(f"{number}\n")
    with open('{}_env_{}_{}_test_acc.txt'.format(args.dataset, \
                str(args.test_envs[0]), args.algorithm), 'w') as file:
        for number in acc_list['test_acc']:
            file.write(f"{number}\n")

    save_checkpoint('model_{}_{}_{}.pkl'.format(args.dataset, str(args.test_envs[0]), args.algorithm))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
