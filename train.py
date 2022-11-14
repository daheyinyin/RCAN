
# ============================================================================
"""train"""
import time
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.communication.management import get_group_size, get_rank
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.args import args
from src.data.div2k import DIV2K
from src.rcan_model import RCAN

def train():
    """train"""
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    if args.run_distribute:
        init()
        context.reset_auto_parallel_context()
        device_num = get_group_size()
        rank_id = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num,
                                          gradients_mean=True)
        if args.device_target == 'Ascend':
            context.set_context(device_id=args.device_id)
    else:
        device_num = 1
        rank_id = 0
        context.set_context(device_id=args.device_id)
    if args.modelArts_mode:
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)

    train_dataset = DIV2K(args, name=args.data_train, train=True, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True, num_parallel_workers=4)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)
    net_m = RCAN(args)
    print("Init net weights successfully")

    if args.ckpt_path:
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")
    step_size = train_de_dataset.get_dataset_size()
    lr = []
    for i in range(0, args.epochs):
        cur_lr = args.lr / (2 ** ((i + 1) // 200))
        lr.extend([cur_lr] * step_size)
    opt = nn.Adam(net_m.trainable_params(), learning_rate=lr, loss_scale=args.loss_scale)
    loss = nn.L1Loss()
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=args.init_loss_scale, \
             scale_factor=2, scale_window=1000)
    model = Model(net_m, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level='auto')
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=args.ckpt_save_interval * step_size,
                                 keep_checkpoint_max=args.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(prefix="rcan", directory=args.ckpt_save_path, config=config_ck)

    if rank_id == 0:
        cb += [ckpt_cb]
    model.train(args.epochs, train_de_dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == "__main__":
    time_start = time.time()
    train()
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))
