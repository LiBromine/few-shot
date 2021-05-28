# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from common.args import parse_args
from common.util import load_few_data, load_large_data, load_test_data
from model import get_model


def shuffle(X, y, shuffle_parts):
    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train_epoch(args, model, X, y, optimizer): # Training Process
    model.train()
    loss, acc = 0.0, 0.0
    num_batches = args.batch_size 
    for i in range(num_batches):
        T, targ = sample_per_class(X)  # (C, u)
        S = sample_statistics(X) # (C, u)
        loss_, acc_ = model(S, T, targ) # (C, u)

        loss_.backward()
        optimizer.step()

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
    loss /= num_batches
    acc /= num_batches
    return acc, loss
    
def valid_epoch(args, model, X, y, alexnet): # Valid Process
    model.eval()
    loss, acc = 0.0, 0.0
    st, ed, times = 0, args.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
        with torch.no_grad(): # errata
            loss_, acc_ = model(X_batch, y_batch)

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()

        st, ed = ed, ed + args.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def inference(model, X): # Test Process
    model.eval()
    with torch.no_grad(): # errata
        pred_ = model(torch.from_numpy(X).to(device))
    return pred_.cpu().data.numpy()


def main_worker(args):
    # init config
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = models.alexnet(pretrained=args.pretrained)

    # Data-loader of val and test set
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.ImageFolder(
        args.few_data_dir,
        transforms,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )


    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if args.is_train:
        X_train, y_train = load_large_data(args.large_data_dir)
        X_val, y_val = load_few_data(args.few_data_dir) # TODO
        mlp_model = get_model()
        mlp_model.to(device)
        print(mlp_model)
        optimizer = optim.Adam(mlp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            
        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        for epoch in range(1, args.num_epochs+1):
            start_time = time.time()
            train_acc, train_loss = train_epoch(args, mlp_model, X_train, y_train, optimizer)

            val_acc, val_loss = valid_epoch(mlp_model, X_val, y_val)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # test_acc, test_loss = valid_epoch(mlp_model, X_test, y_test)
                with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(epoch)), 'wb') as fout:
                    torch.save(mlp_model, fout)
                with open(os.path.join(args.train_dir, 'checkpoint_0.pth.tar'), 'wb') as fout:
                    torch.save(mlp_model, fout)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            # print("  test loss:                     " + str(test_loss))
            # print("  test accuracy:                 " + str(test_acc))

            if train_loss > max(pre_losses):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9995
            pre_losses = pre_losses[1:] + [train_loss]

    else:
        mlp_model = get_model()
        mlp_model.to(device)
        model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
        if os.path.exists(model_path):
            mlp_model = torch.load(model_path)

        X_test = load_test_data(args.test_data_dir)

        count = 0
        # for i in range(len(X_test)):
        #     test_image = X_test[i].reshape((1, 3 * 32 * 32))
        #     result = inference(mlp_model, test_image)[0]
        #     if result == y_test[i]:
        #         count += 1
        print("test accuracy: {}".format(float(count) / len(X_test)))


if __name__ == '__main__':
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    main_worker(args)
