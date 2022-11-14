
# ============================================================================
"""args"""
import argparse
import ast

parser = argparse.ArgumentParser(description='RCAN')

# Hardware specifications
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/cache/data/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default=2,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='RCAN',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')


# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                    help="Run distribute, default is false.")
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='device target, Ascend or GPU (Default: Ascend)')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--test_every', type=int, default=4000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--loss_scale', type=float, default=1024.0,
                    help='scaling factor for optim')
parser.add_argument('--init_loss_scale', type=float, default=65536.,
                    help='scaling factor')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# ckpt specifications
parser.add_argument('--ckpt_save_path', type=str, default='./ckpt/',
                    help='path to save ckpt')
parser.add_argument('--ckpt_save_interval', type=int, default=10,
                    help='save ckpt frequency, unit is epoch')
parser.add_argument('--ckpt_save_max', type=int, default=10,
                    help='max number of saved ckpt')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='path of saved ckpt')

# ModelArts
parser.add_argument('--modelArts_mode', type=ast.literal_eval, default=False,
                    help='train on modelarts or not, default is False')
parser.add_argument('--data_url', type=str, default='', help='the directory path of saved file')


args, unparsed = parser.parse_known_args()

args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 100

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
