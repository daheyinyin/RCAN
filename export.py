
# ============================================================================
"""export net together with checkpoint into air/mindir/onnx models"""
import os
import argparse
import numpy as np
from src.rcan_model import RCAN
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, export


parser = argparse.ArgumentParser(description='rcan export')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_path", type=str, required=True, help="path of checkpoint file")
parser.add_argument("--file_name", type=str, default="rcan", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
parser.add_argument('--scale', type=int, default='2', help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
args_1 = parser.parse_args()

MAX_HR_SIZE = 2040

def run_export(args):
    """ export """
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    net = RCAN(args)
    max_lr_size = MAX_HR_SIZE // args.scale  #max_lr_size = MAX_HR_SIZE / scale
    param_dict = load_checkpoint(args.ckpt_path)
    net.load_pre_trained_param_dict(param_dict, strict=False)
    net.set_train(False)
    print('load mindspore net and checkpoint successfully.')
    inputs = Tensor(np.ones([args.batch_size, 3, max_lr_size, max_lr_size]), ms.float32)
    export(net, inputs, file_name=args.file_name, file_format=args.file_format)
    print('export successfully!')


if __name__ == "__main__":
    run_export(args_1)
