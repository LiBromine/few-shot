import numpy as np 
import matplotlib.pyplot as plt
import os

names = ['exp_eye', 'exp_linear', 'exp_nonlinear']

def plot_loss_train():

    loss_linear = np.load('./exp_linear/log/train_loss_epoch.npy')
    loss_unlinear = np.load('./exp_nonlinear/log/train_loss_epoch.npy')
    loss_eye = np.load('./exp_eye/log/train_loss_epoch.npy')

    # iter1 = np.arange(loss_linear.shape[0])
    # iter2 = np.arange(loss_.shape[0])
    plt.figure()
    plt.subplot(111)
    plt.plot(np.arange(loss_linear.shape[0]), loss_linear, '-', label='Linear')
    plt.plot(np.arange(loss_unlinear.shape[0]), loss_unlinear, '-', label='non-Linear')
    plt.plot(np.arange(loss_eye.shape[0]), loss_eye, '-', label='Identity Init')
    # plt.plot(iter_, loss_lge, '-', label='EuclideanLoss')
    # plt.plot(iter_, loss_lgs, '-', label='SoftmaxCrossEntropyLoss')
    # plt.plot(iter_, loss_lgh, '-', label='HingeLoss')
    plt.xlabel(r'# of Epoches')
    plt.ylabel(r'Loss')
    plt.title(r'Training loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("./loss_train.png")

def plot_acc_train():
    acc_linear = np.load('./exp_linear/log/train_acc_epoch.npy')
    acc_unlinear = np.load('./exp_nonlinear/log/train_acc_epoch.npy')
    acc_eye = np.load('./exp_eye/log/train_acc_epoch.npy')

    # iter_ = np.arange(acc_l.shape[0]) * 50
    plt.figure()
    plt.subplot(111)
    plt.plot(np.arange(acc_linear.shape[0]), acc_linear, '-', label='Linear')
    plt.plot(np.arange(acc_unlinear.shape[0]), acc_unlinear, '-', label='non-Linear')
    plt.plot(np.arange(acc_eye.shape[0]), acc_eye, '-', label='Identity Init')
    # plt.plot(iter_, acc_lge, '-', label='EuclideanLoss')
    # plt.plot(iter_, acc_lgs, '-', label='SoftmaxCrossEntropyLoss')
    # plt.plot(iter_, acc_lgh, '-', label='HingeLoss')
    plt.xlabel(r'# of Epoches')
    plt.ylabel(r'Accuracy')
    plt.title(r'Training Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("./acc_train.png")

def plot_loss_test():
    loss_linear = np.load('./exp_linear/log/val_loss_epoch.npy')
    loss_unlinear = np.load('./exp_nonlinear/log/val_loss_epoch.npy')
    loss_eye = np.load('./exp_eye/log/val_loss_epoch.npy')

    plt.figure()
    plt.subplot(111)
    plt.plot(np.arange(loss_linear.shape[0]), loss_linear, '-', label='Linear')
    plt.plot(np.arange(loss_unlinear.shape[0]), loss_unlinear, '-', label='non-Linear')
    plt.plot(np.arange(loss_eye.shape[0]), loss_eye, '-', label='Identity Init')
    # plt.plot(iter_, loss_lge, '-', label='EuclideanLoss')
    # plt.plot(iter_, loss_lgs, '-', label='SoftmaxCrossEntropyLoss')
    # plt.plot(iter_, loss_lgh, '-', label='HingeLoss')
    plt.xlabel(r'# of Epoches')
    plt.ylabel(r'Loss')
    plt.title(r'Validation loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("./loss_val.png")

def plot_acc_test():
    acc_linear = np.load('./exp_linear/log/val_acc_epoch.npy')
    acc_unlinear = np.load('./exp_nonlinear/log/val_acc_epoch.npy')
    acc_eye = np.load('./exp_eye/log/val_acc_epoch.npy')

    # iter_ = np.arange(acc_l.shape[0]) * 50
    plt.figure()
    plt.subplot(111)
    plt.plot(np.arange(acc_linear.shape[0]), acc_linear, '-', label='Linear')
    plt.plot(np.arange(acc_unlinear.shape[0]), acc_unlinear, '-', label='non-Linear')
    plt.plot(np.arange(acc_eye.shape[0]), acc_eye, '-', label='Identity Init')
    # plt.plot(iter_, acc_lge, '-', label='EuclideanLoss')
    # plt.plot(iter_, acc_lgs, '-', label='SoftmaxCrossEntropyLoss')
    # plt.plot(iter_, acc_lgh, '-', label='HingeLoss')
    plt.xlabel(r'# of Epoches')
    plt.ylabel(r'Accuracy')
    plt.title(r'Validation Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("./acc_val.png")

def find_max(exp_name):
    path = os.path.join(exp_name, 'log')
    path = os.path.join(path, 'val_acc_epoch.npy')
    with open(path, 'rb') as f:
        data = np.load(f)
        max_acc = np.max(data)
        epoch = np.argmax(data)
        print('{} with best acc {}, at {} epoch'.format(exp_name, max_acc, epoch))

if __name__ == "__main__":
    # plot_loss_train()
    # plot_acc_train()
    # plot_loss_test()
    # plot_acc_test()
    for name in names:
        find_max(name)