import time
import argparse

from pathlib import Path

import numpy as np
import torch
import tqdm

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass

import cbs2.bird_view.utils.bz_utils as bzu

from cbs2.bird_view.models.birdview import BirdViewPolicyModelSS
from cbs2.bird_view.models.image import ImagePolicyModelSS
from cbs2.bird_view.utils.train_util import one_hot
#from utils.image_utils import draw_msra_gaussian, gaussian_radius, CoordinateConverter
from cbs2.bird_view.utils.datasets.image_lmdb import get_image as load_data
from cbs2.bird_view.utils.datasets.birdview_lmdb import Location, Transform, Rotation

BACKBONE = 'resnet34'
# BACKBONE = 'resnet50'
GAP = 5
N_STEP = 5
PIXELS_PER_METER = 5
CROP_SIZE = 192
SAVE_EPOCHS = range(0, 1000, 2)


class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=2.0, device='cuda'):
        self._img_size = torch.FloatTensor([w,h]).to(device)

        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset

    def __call__(self, camera_locations):
        camera_locations = (camera_locations + 1) * self._img_size/2
        w, h = self._img_size

        cx, cy = w/2, h/2

        f = w /(2 * np.tan(self._fov * np.pi / 360))

        xt = (camera_locations[...,0] - cx) / f
        yt = (camera_locations[...,1] - cy) / f

        world_z = self._world_y / yt
        world_x = world_z * xt

        map_output = torch.stack([world_x, world_z],dim=-1)

        map_output *= PIXELS_PER_METER
        map_output[...,1] = CROP_SIZE - map_output[...,1]
        map_output[...,0] += CROP_SIZE/2
        map_output[...,1] += self._fixed_offset*PIXELS_PER_METER

        return map_output


class LocationLoss(torch.nn.Module):
    def __init__(self, w=384, h=160, device='cuda', **kwargs):
        super().__init__()
        self._img_size = torch.FloatTensor([w,h]).to(device)

    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations/(0.5*self._img_size) - 1
        teac_locations = teac_locations / (0.5 * self._img_size) - 1

        #return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1,2,3)) #L1
        return torch.mean(torch.pow((pred_locations - teac_locations),2), dim=(1,2,3)) #L2 TEMPORARY TEST - 06/01/2022

def _log_visuals(rgb_image, birdview, speed, command, loss, _pred_locations, _teac_locations, wp_method, size=8):
    import cv2
    import numpy as np
    import cbs2.bird_view.utils.carla_utils as cu

    WHITE = [255, 255, 255]
    GREEN = [0, 255, 0] # Teacher pred (supervision)
    ORANGE = [255, 100, 0] # Student pred
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = list()

    #for i in range(min(birdview.shape[0],size)):
    for i in range(min(birdview.shape[0],size)): # TEMPORARY SHOWS THE ONE WITH THE BIGGER LOSS
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rgb = np.uint8(_numpy(rgb_image[i]).transpose(1, 2, 0) * 255).copy()
        rgb = rgb[:, :, [2, 1, 0]] # BGR to RGB
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

        def _write(text, i, j):
            cv2.putText(
                    canvas, text, (cols[j], rows[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        def _dot(_canvas, i, j, color, radius=2):
            x, y = int(j), int(i)
            _canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color

        def _stick_together(a, b):
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 1)

        _command = {
                1: 'LEFT', 2: 'RIGHT',
                3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item()+1, '???')

        _wp_method = {
            0: 'InFrame', 1: 'Intrpl', 2: '<2',
            3: 'Stop(TL)', 4: 'Stop(Obs)'}.get(
            torch.argmax(wp_method[i]).item(), '???')

        _dot(canvas, 0, 0, WHITE)

        for x, y in _teac_locations[i]: _dot(canvas, x, y, GREEN)
        for x, y in _teac_locations[i]: _dot(rgb, x, y, GREEN)
        for x, y in _pred_locations[i]: _dot(rgb, x, y, ORANGE)
        for x, y in _pred_locations[i]: _dot(canvas, x, y, ORANGE)

        _write('%s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)
        _write('%s' % _wp_method, 3, 0)
        _write('Spd: %.2f' % np.abs(speed[i].item()), 4, 0)

        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    #return [x[1] for x in images]
    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])] # Plot the ones with bigger loss


def repeat(a, repeats, dim=0):
    """
    Substitute for numpy's repeat function. Taken from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)

def unproject(output, img_size, fixed_offset=0., world_y=1.4, fov=90):
    cx, cy = img_size / 2

    w, h = img_size

    f = w / (2 * np.tan(fov * np.pi / 360))

    xt = (output[..., 0:1] - cx) / f
    yt = (output[..., 1:2] - cy) / f

    world_z = world_y / yt
    world_x = world_z * xt

    world_output = np.stack([world_x, world_z], axis=-1)

    if fixed_offset:
        world_output[..., 1] -= fixed_offset

    world_output = world_output.squeeze()

    return world_output

def train_or_eval(coord_converter, coord_converter_teach, criterion, net,
                  teacher_net, data, optim, is_train, config, is_first_epoch):
    if is_train:
        desc = 'Train'
        net.train()
    else:
        desc = 'Val'
        net.eval()

    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)

    tick = time.time()

    import torch.distributions as tdist
    noiser = tdist.Normal(torch.tensor(0.0), torch.tensor(config['speed_noise']))

    for i, (rgb_image, birdview, location, command, speed, wp_method) in iterator:
        rgb_image = rgb_image.to(config['device'])
        birdview = birdview.to(config['device'])
        command = one_hot(command).to(config['device'])
        speed = speed.to(config['device'])

        # print('\nBV: {}'.format(birdview.shape))
        # print('RGB: {}'.format(rgb_image.shape))
        # print('Speed: {}'.format(speed.shape))
        # print('Cmd: {}'.format(command.shape))

        if is_train and config['speed_noise'] > 0:
            speed += noiser.sample(speed.size()).to(speed.device)
            speed = torch.clamp(speed, 0, 10)

        if len(rgb_image.size()) > 4:
            B, batch_aug, c, h, w = rgb_image.size()
            rgb_image = rgb_image.view(B*batch_aug,c,h,w)
            birdview = repeat(birdview, batch_aug)
            command = repeat(command, batch_aug)
            speed = repeat(speed, batch_aug)

        with torch.no_grad():
            _teac_location, _teac_locations = teacher_net(birdview, speed, command)

        #summary(net, input_size=[rgb_image.shape[1:], speed.shape[1:], command.shape[1:]])
        _pred_location, _pred_locations = net(rgb_image, speed, command)
        _pred_locations = (_pred_locations + 1) * coord_converter._img_size / 2
        _teac_locations = (_teac_locations + 1) * coord_converter._img_size / 2

        loss = criterion(_pred_locations, _teac_locations)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        should_log = False
        should_log |= i % config['log_iterations'] == 0
        should_log |= not is_train
        should_log |= is_first_epoch

        if should_log:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = _log_visuals(
                    rgb_image, birdview, speed, command, loss,
                    (_pred_location+1)*coord_converter._img_size/2, (_teac_location+1)*coord_converter._img_size/2, wp_method)

            bzu.log.scalar(is_train=is_train, loss_mean=loss_mean.item())
            bzu.log.image(is_train=is_train, birdview=images)

        bzu.log.scalar(is_train=is_train, fps=1.0/(time.time() - tick))

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break


def train(config):
    bzu.log.init(config['log_dir'])
    bzu.log.save_config(config)
    teacher_config = bzu.log.load_config(config['teacher_args']['model_path'])

    # If teacher used combined segmentation, add it to config
    if teacher_config['data_args']['combine_seg']:
        config['data_args'] = True

    data_train, data_val = load_data(**config['data_args'])
    criterion = LocationLoss()
    assert not(config['model_args']['fpn'] and (config['model_args']['ppm_bins'] is not None)), 'FPN and PPM cannot be used simultaneously'
    net = ImagePolicyModelSS(
        config['model_args']['backbone'],
        pretrained=config['model_args']['imagenet_pretrained'],
        ppm_bins = config['model_args']['ppm_bins'],
        fpn = config['model_args']['fpn'],
        all_branch=True
    ).to(config['device'])
    net.load_state_dict(torch.load(config['phase0_ckpt']))

    teacher_net = BirdViewPolicyModelSS(teacher_config['model_args']['backbone'],
                                        pretrained=True,
                                        all_branch=True,
                                        input_channel=teacher_config['model_args']['input_channel'],
                                        seg=teacher_config['data_args']['segmentation']).to(config['device'])
    teacher_net.load_state_dict(torch.load(config['teacher_args']['model_path']))
    teacher_net.eval()

    coord_converter = CoordConverter(**config['agent_args']['camera_args'])
    coord_converter_teach = CoordConverter(**config['teacher_args']['camera_args'])

    optim = torch.optim.Adam(net.parameters(), lr=config['optimizer_args']['lr'])

    for epoch in tqdm.tqdm(range(int(config['max_epoch'])+1), desc='Epoch'):
        train_or_eval(coord_converter, coord_converter_teach, criterion, net,teacher_net, data_train, optim, True, config, epoch == 0)
        train_or_eval(coord_converter, coord_converter_teach, criterion, net,teacher_net, data_val, None, False, config, epoch == 0)

        if epoch in SAVE_EPOCHS:
            torch.save(
                    net.state_dict(),
                    str(Path(config['log_dir']) / ('model-%d.th' % epoch)))

        bzu.log.end_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=1000)
    parser.add_argument('--max_epoch', default=256)

    # Model
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ckpt', required=True)

    # Teacher.
    parser.add_argument('--teacher_path', required=True)

    parser.add_argument('--fixed_offset', type=float, default=4.)

    # Dataset.
    parser.add_argument('--batch_aug', type=int, default=1)
    parser.add_argument('--dataset_dir', default='/raid0/dian/carla_0.9.6_data')
    #parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32) # For ResNet50 to avoid CUDA out of memory
    parser.add_argument('--speed_noise', type=float, default=0.1)
    parser.add_argument('--augment', choices=['medium', 'medium_harder', 'super_hard', 'None', 'custom'], default='super_hard')

    # Student network Pyramid Pooling Module / Feature Pyramid Network
    parser.add_argument('--ppm', default="")
    parser.add_argument('--fpn', action='store_true')

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)

    parsed = parser.parse_args()

    config = {
            'log_dir': parsed.log_dir,
            'log_iterations': parsed.log_iterations,
            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'phase0_ckpt': parsed.ckpt,
            'optimizer_args': {'lr': parsed.lr},
            'speed_noise': parsed.speed_noise,
            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                'n_step': N_STEP,
                'gap': GAP,
                'augment': parsed.augment,
                'batch_aug': parsed.batch_aug,
                'num_workers': 16,
                },
            'model_args': {
                'model': 'image_ss',
                'imagenet_pretrained': parsed.pretrained,
                'backbone': BACKBONE,
                'fpn': parsed.fpn,
                'ppm_bins': None if len(parsed.ppm)==0 else list(map(int, parsed.ppm.split("-"))), # "1-2-3-6" --> [1, 2, 3, 6]
                },
            'teacher_args' : {
                'model_path': parsed.teacher_path,
                'input_channel': 13,
                'camera_args': {
                    'w': 384,
                    'h': 160,
                    'fov': 120,
                    'world_y': 1.4,
                    'fixed_offset': 4.0,
                },
                },
            'agent_args': {
                'camera_args': {
                    'w': 384,
                    'h': 160,
                    'fov': 90,
                    'world_y': 1.4,
                    'fixed_offset': 4.0,
                },
            }
        }

    train(config)
