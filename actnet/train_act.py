# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from common.args import parse_args
from common.util import load_large_data
from model import get_model

def sample_per_class(X, y):
    '''
    X, Tensor (cls, num_sample, num_features/units)
    y, Tensor (cls,)
    '''
    cls_num = X.shape[0]
    num_sample = X.shape[1]
    data_list = []
    for i in range(cls_num):
        sample = X[i, np.random.randint(num_sample)]
        data_list.append(sample.reshape(1, -1))
    X_hat = torch.cat(data_list, dim=0) # (C, u)
    
    if y is not None:
        # shuffle
        index = np.arange(0, cls_num)
        np.random.shuffle(index)
        return X_hat[index].to(device), y[index].to(device)
    else:
        # non shuffle version
        return X_hat, y


def sample_statistics(X, y, p_mean=0.95):
    X = torch.as_tensor(X)

    cls_num = X.shape[0]
    X_mean = X.mean(dim=1) # (C, u)
    X_hat, _ = sample_per_class(X, None) # (C, u)
    proba = torch.ones(X.shape[0]) * p_mean
    sample = torch.bernoulli(proba)
    for i in range(len(sample)):
        if sample[i] < 0.5:
            X_mean[i] = X_hat[i]

    data = torch.zeros_like(X_mean)
    label = torch.zeros_like(y)
    for i in range(len(sample)):
        data[y[i].int()] = X_mean[i]

    # print('S label: ', label)
    # index = np.arange(0, cls_num)
    # np.random.shuffle(index)
    return data.to(device)


def train_epoch(args, model, X, y, optimizer): # Training Process
    model.train()
    loss, acc = 0.0, 0.0
    losses = []
    acces = []
    for i in range(args.num_batches):
        optimizer.zero_grad()
        T, targ = sample_per_class(X, y)  # (C, u)
        S = sample_statistics(X, y, args.p_mean) # (C, u)
        loss_, acc_ = model(S, T, targ) # (C, u)

        loss_.backward()
        optimizer.step()

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
        losses.append(loss_.cpu().data.numpy())
        acces.append(acc_.cpu().data.numpy())

    loss /= args.num_batches
    acc /= args.num_batches
    return acc, loss, acces, losses


def valid_epoch(args, model, alexnet, loader): # Valid Process
    model.eval()
    loss, acc = 0.0, 0.0
    num_test = args.num_shots - args.val_num_shots
    for i, (images, target) in enumerate(loader):
        assert i == 0
        with torch.no_grad():
            target = target.to(device)
            features = alexnet(images) # (Big_B, u)
            assert features.shape[0] % args.num_shots == 0
            features = features.reshape(-1, args.num_shots, args.num_units) # (C, shot, u)
            features = features.to(device)
            target = target.reshape(-1, args.num_shots)[:, :num_test].flatten() # (class_num * num_test, )
            shoters = features[:, :args.val_num_shots].to(device) # (class_num, vns, u)
            testers = features[:, args.val_num_shots:].reshape(-1, args.num_units).to(device) # (class_num, vns, u) -> # (cls_num * vns, u)    
            loss_, acc_ = model(shoters, testers, target, few=True, shot=args.val_num_shots)

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()

    return acc, loss


def inference(model, X): # Test Process, TODO
    model.eval()
    with torch.no_grad(): # errata
        pred_ = model(torch.from_numpy(X).to(device))
    return pred_.cpu().data.numpy()


def main_worker(args):
    # init config
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    alexnet = models.alexnet(pretrained=args.pretrained)
    new_classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
    alexnet.classifier = new_classifier
    for p in alexnet.parameters():
        p.requires_grad = False
    
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
        transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=500, shuffle=False,
        pin_memory=True
    )

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if args.is_train:
        X_train, y_train = load_large_data(args.large_data_dir)

        train_loss_batch = np.array([], dtype=np.float32)
        train_acc_batch = np.array([], dtype=np.float32)
        train_loss_epoch = np.array([], dtype=np.float32)
        train_acc_epoch = np.array([], dtype=np.float32)
        val_loss_epoch = np.array([], dtype=np.float32)
        val_acc_epoch = np.array([], dtype=np.float32)

        mlp_model = get_model(num_units=args.num_units)
        mlp_model.to(device)
        print(mlp_model)
        batch_size = X_train.shape[0] # 1000
        print('batch_size: ', batch_size)
        optimizer = optim.SGD(mlp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
            
        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        print('Start iterations...')
        for epoch in range(1, args.num_epochs+1):
            start_time = time.time()
            train_acc, train_loss, batches_acc, batches_loss = train_epoch(args, mlp_model, X_train, y_train, optimizer)

            val_acc, val_loss = valid_epoch(args, mlp_model, alexnet, val_loader)

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

            train_loss_epoch = np.append(train_loss_epoch, train_loss)
            train_acc_epoch = np.append(train_acc_epoch, train_acc)
            val_loss_epoch = np.append(val_loss_epoch, val_loss)
            val_acc_epoch = np.append(val_acc_epoch, val_acc)
            train_loss_batch = np.append(train_loss_batch, np.array(batches_loss))
            train_acc_batch = np.append(train_acc_batch, np.array(batches_acc))

            if train_loss > max(pre_losses):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9995
            pre_losses = pre_losses[1:] + [train_loss]

        filelist = ['train_loss_batch.npy', 'train_loss_epoch.npy', 'val_loss_epoch.npy',
            'train_acc_batch.npy', 'train_acc_epoch.npy', 'val_acc_epoch.npy']
        np.save(os.path.join(args.log_dir, filelist[0]), train_loss_batch)
        np.save(os.path.join(args.log_dir, filelist[1]), train_loss_epoch)
        np.save(os.path.join(args.log_dir, filelist[2]), val_loss_epoch)
        np.save(os.path.join(args.log_dir, filelist[3]), train_acc_batch)
        np.save(os.path.join(args.log_dir, filelist[4]), train_acc_epoch)
        np.save(os.path.join(args.log_dir, filelist[5]), val_acc_epoch)

    else:
        mlp_model = get_model()
        mlp_model.to(device)
        # TODO
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
