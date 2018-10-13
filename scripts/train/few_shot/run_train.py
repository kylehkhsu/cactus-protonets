import argparse

from train import main
import ipdb
import os

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'omniglot'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
parser.add_argument('--data.rotations', action='store_true', help="3x data augmentation via rotations for Omniglot")
parser.add_argument('--data.encoder', type=str, default='none', help='encoder name')
parser.add_argument('--data.train_mode', type=str, default='kmeans')
parser.add_argument('--data.test_mode', type=str, default='ground_truth')

parser.add_argument('--data.clusters', type=int, default=500)
parser.add_argument('--data.partitions', type=int, default=100)

# model args
default_model_name = 'protonet_conv'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,28,28', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=32, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 32)")

# train args
parser.add_argument('--train.epochs', type=int, default=300, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
parser.add_argument('--log.date', type=str, default='19991231')
parser.add_argument('--log.suffix', type=str, default='')

# args = vars(parser.parse_args())

opt = vars(parser.parse_args())

exp_str = ''

if opt['data.train_mode'] == 'ground_truth':
    exp_str += '_oracle'
elif opt['data.train_mode'] == 'random':
    exp_str += '_random'
elif opt['data.train_mode'] == 'kmeans':
    exp_str += '_{}'.format(opt['data.encoder'])

if opt['data.train_mode'] == 'kmeans' or opt['data.train_mode'] == 'random':
    exp_str += '_k{}_p{}'.format(opt['data.clusters'], opt['data.partitions'])

exp_str += '_way{}_shot{}_query{}_hdim{}'.format(opt['data.way'], opt['data.shot'], opt['data.query'], opt['model.hid_dim'])
if opt['log.suffix'] != '':
    exp_str += '_{}'.format(opt['log.suffix'])
exp_str = exp_str[1:]
exp_str = os.path.join(opt['data.dataset'], opt['log.date'], exp_str)
opt['log.exp_dir'] = os.path.join(opt['log.exp_dir'], exp_str)
print(opt['log.exp_dir'])

main(opt)
