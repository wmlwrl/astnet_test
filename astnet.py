import pprint
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
import datasets
import models as models
from utils import utils
from config import config, update_config
import core
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet for Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)
    return args

def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']
    inputs = video[:-1]
    target = video[-1]
    return inputs, target

class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)

def main():
    intensity_loss = Intensity_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()

    fig = plt.figure("Image")
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry = (550, 200, 600, 500)
    # This works for QT backend, for other backends, check this ⬃⬃⬃.
    # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib

    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)
    model = models.get_net(config)

    logger.info('Model: {}'.format(model.get_name()))
    model=model.cuda()

    #model = nn.DataParallel(model, device_ids=gpus).cuda()
    #model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    mat_loader = datasets.get_label(config)
    mat = mat_loader()
    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    #model.module.load_state_dict(state_dict)

    psnr_list = core.inference(config, test_loader, model)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, _, _ = utils.calculate_auc(config, psnr_list, mat)

    logger.info(f'AUC: {auc * 100:.1f}%' )


if __name__ == '__main__':
    main()
