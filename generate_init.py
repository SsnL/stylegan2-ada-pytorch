# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import math
import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import PIL

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_sampling_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    init_seed  = None, # Random seed: <int>, default = 0
    out_subdir = None, # <str>

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    nchannels  = None, # num channels: <int>
    res        = None, # resolution: <int>

    # Sampling options.
    sampling_seeds = None, # <List[int]>
    truncation_psi = None, # <float>
    noise_mode = None, # <str>
    reinit_sample_interval = None, # <int>
    save_ty = None, # <str>

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    args.num_gpus = gpus

    if init_seed is None:
        init_seed = 0
    assert isinstance(init_seed, int)
    args.init_seed = init_seed

    # ------------------------------------
    # Base config: cfg, gamma
    # ------------------------------------

    if cfg is None:
        cfg = 'stylegan2'
    assert isinstance(cfg, str)
    desc = f'{cfg}-nc{nchannels}-res{res}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'-gpus{gpus:d}'
        spec.ref_gpus = gpus
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow

    args.G_kwargs.c_dim = 0
    args.G_kwargs.img_channels = nchannels
    args.G_kwargs.img_resolution = res

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing

    # -------------------------------------------------
    # Sampling options
    # -------------------------------------------------
    args.sampling_seeds = sampling_seeds
    args.truncation_psi = truncation_psi
    args.noise_mode = noise_mode
    if reinit_sample_interval <= 0:
        reinit_sample_interval = None
    args.reinit_sample_interval = reinit_sample_interval
    args.save_ty = save_ty
    desc += f'-n{len(sampling_seeds)}-psi{truncation_psi:g}-noise{noise_mode}'
    if '=>' in save_ty:
        desc += '-stats'
    else:
        desc += f'-{save_ty}'
    if args.reinit_sample_interval is not None:
        desc += f'-reinit{reinit_sample_interval}'

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if out_subdir is not None:
        desc += f"/{out_subdir}"

    return desc, args

#----------------------------------------------------------------------------

import contextlib
import numpy as np


@contextlib.contextmanager
def with_seed(seed):
    npstate = np.random.get_state(seed)
    thstate = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(npstate)
        torch.set_rng_state(thstate)


from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix


class GManager(object):
    def __init__(self, G_kwargs, device, init_base_seed, reinit_interval):
        self.G_kwargs = G_kwargs
        self.device = device
        self.init_base_seed = init_base_seed
        self.reinit_interval = reinit_interval
        self.G_seed = None
        torch.backends.cudnn.benchmark = True    # Improves training speed.
        conv2d_gradfix.enabled = True                       # Improves training speed.

    def init_G(self, seed):
        with with_seed(seed):
            self.G = dnnlib.util.construct_class_by_name(
                **self.G_kwargs).eval().requires_grad_(False).to(self.device) # subclass of torch.nn.Module
            self.G_seed = seed

    def get_G_seed(self, idx):
        if self.reinit_interval is None:
            return self.init_base_seed
        else:
            return self.init_base_seed + (idx // self.reinit_interval)

    def get_G(self, idx):
        desired_seed = self.get_G_seed(idx)
        if self.G_seed != desired_seed:
            self.init_G(desired_seed)
        return self.G


class Saver(object):
    def __init__(self, ty, save_dir):
        self.ty = ty
        self.save_dir = save_dir
        if ty not in {'raw_tensor', 'min_max_unnorm', 'pm1_unnorm'}:
            import scipy.io
            outstats, tgtstats = ty.split('=>')
            self.output_stats = {
                    k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in scipy.io.loadmat(outstats).items()
            }
            self.target_stats = {
                    k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in scipy.io.loadmat(tgtstats).items()
            }

    def save(self, output_tensor, idx):
        save_file_wo_ext = os.path.join(self.save_dir, f'{idx:06d}')
        if self.ty == 'raw_tensor':
            torch.save(output_tensor.cpu(), save_file_wo_ext + '.pth')
            return

        img = output_tensor
        if self.ty == 'min_max_unnorm':
            img -= img.flatten(1, 2).min(dim=1).values[:, None, None]
            img /= img.flatten(1, 2).max(dim=1).values[:, None, None]
            img = (img.permute(1, 2, 0) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
        elif self.ty == 'pm1_unnorm':
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        else:
            img = img.permute(1, 2, 0).flatten(0, 1)
            img = img.cpu()
            img = (img - self.output_stats['mu']) @ self.output_stats['w']
            img = img @ self.target_stats['winv'] + self.target_stats['mu']
            img = img.reshape(*output_tensor.shape[1:], output_tensor.shape[0])
            img = (img * 255 + 0.5).clamp(0, 255).to(torch.uint8)

        PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(save_file_wo_ext + '.png')

def sample(G_kwargs, init_seed,
           noise_mode, num_gpus, reinit_sample_interval, run_dir,
           sampling_seeds, truncation_psi, save_ty, *, rank):
    device = torch.device('cuda', rank)
    G_manager = GManager(
        G_kwargs, device,
        init_base_seed=init_seed,
        reinit_interval=reinit_sample_interval)
    saver = Saver(save_ty, run_dir)

    label = torch.zeros([1, 0], device=device)

    for seed_idx, seed in sampling_seeds.get_split(num_gpus, rank=rank).iter_enumerate():
        G = G_manager.get_G(seed_idx)
        assert G.c_dim == 0

        print(f'Generating seed={seed} ({seed_idx}/{len(sampling_seeds)}) w/ G_seed={G_manager.G_seed} ...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)[0].detach()
        saver.save(img, seed_idx)


#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    sample(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------


class num_range:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    def __init__(self, s: str, num_splits=1, rank=0):
        self.desc = s
        self.num_splits = num_splits
        self.rank = rank

        range_re = re.compile(r'^(\d+)-(\d+)$')
        m = range_re.match(s)
        if m:
            vals = list(range(int(m.group(1)), int(m.group(2))+1))
        else:
            vals = s.split(',')
            vals = [int(x) for x in vals]

        nv_per_split = int(math.ceil(len(vals) / num_splits))
        self.total_len = len(vals)
        self.start = nv_per_split * rank
        end = min(len(vals), nv_per_split * (rank + 1))
        self.vals = vals[self.start: end]

    def get_split(self, num_splits, rank):
        if self.num_splits == 1 and self.rank == 0:
            return self.__class__(self.desc, num_splits=num_splits, rank=rank)
        else:
            raise RuntimeError()

    def iter_enumerate(self):
        yield from enumerate(self.vals, start=self.start)

    def __iter__(self):
        yield from self.vals

    def __len__(self):
        return len(self.vals)

    def len_across_all_splits(self):
        return self.total_len

    def __str__(self):
        return self.desc

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.desc}', num_splits={self.num_splits}, rank={self.rank})"


@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--no-auto-outdir-folder', help='Skip automatically adding a folder under outdir as output, but use outdir directly', is_flag=True)
@click.option('--out_subdir')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--init_seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Base config.
@click.option('--cfg', help='Base config [default: stylegan2]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
@click.option('--nc', 'nchannels', type=int, default=3, metavar='INT')
@click.option('--res', type=int, required=True, metavar='INT')

# Sampling options.
@click.option('--sampling_seeds', type=num_range, required=True, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--reinit-sample-interval', type=int, required=True, default=-1)
@click.option('--save_ty', type=str, default='pm1_unnorm')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')

def main(ctx, outdir, no_auto_outdir_folder, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup options.
    try:
        run_desc, args = setup_sampling_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    if no_auto_outdir_folder:
        args.run_dir = outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
        assert not os.path.exists(args.run_dir)

    # Print options.
    import pprint
    print()
    print('Training options:')
    print(pprint.pformat(args))
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'sampling_options.log'), 'wt') as f:
        print(pprint.pformat(args), file=f)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
