# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
from datetime import datetime
import numpy as np

@HOOKS.register_module()
class TensorboardCustomLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardCustomLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(TensorboardCustomLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        # trick to handle predimg from EvalHook
        valimg = runner.log_buffer.output.get('valimg')
        if valimg is not None:
            # log_buffer is only for scalar items
            del runner.log_buffer.output['valimg']
            self.writer.add_image(f'val/image_{self.get_iter(runner)}', valimg, self.get_iter(runner), dataformats='HWC')
            del valimg

        tags = self.get_loggable_tags(runner, allow_text=True)
        img = runner.outputs.get('img')
        if img is not None:
            del runner.outputs['img']
            if (self.get_iter(runner) % 2000 == 0):
                self.writer.add_image(f'train/image_{self.get_iter(runner)}', img, self.get_iter(runner), dataformats='HWC')
            del img
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()