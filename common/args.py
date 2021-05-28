import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=100,
        help='Batch size for mini-batch training and evaluating. Default: 100')
    parser.add_argument('--num_epochs', type=int, default=20,
        help='Number of training epoch. Default: 20')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
        help='Learning rate during optimization. Default: 1e-3')
    parser.add_argument('--drop_rate', type=float, default=0.5,
        help='Drop rate of the Dropout Layer. Default: 0.5')
    parser.add_argument('--is_train', type=bool, default=True,
        help='True to train and False to inference. Default: True')
    parser.add_argument('--data_dir', type=str, default='../cifar-10_data',
        help='Data directory. Default: ../cifar-10_data')
    parser.add_argument('--train_dir', type=str, default='./train',
        help='Training directory for saving model. Default: ./train')
    parser.add_argument('--inference_version', type=int, default=0,
        help='The version for inference. Set 0 to use latest checkpoint. Default: 0')
        
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
