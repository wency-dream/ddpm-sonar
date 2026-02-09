from model.ddpm import Unet
from model.iddpm import Unet_iddpm
from ddpm_model import GaussianDiffusion
from iddpm_model import LearnedGaussianDiffusion
from trianer import Trainer
import argparse

def get_arg():
    parser = argparse.ArgumentParser(description='PyTorch DDPM Example')
    parser.add_argument('--dim', type=int, default=64,
                             help='base dimension of UNet')
    parser.add_argument('--model_name',type=str,default='DDPM')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=(1, 2, 4, 8),
                             help='dimension multipliers for UNet layers')
    parser.add_argument('--num_classes', type=int, default=5,
                             help='number of classes for conditioning')
    parser.add_argument('--cond_drop_prob', type=float, default=0.5,
                             help='conditional dropout probability')
    parser.add_argument('--image_size', type=int, default=256,
                             help='size of input images')
    parser.add_argument('--timesteps', type=int, default=10,
                             help='number of diffusion timesteps')
    parser.add_argument('--data_dir', type=str, default='E:\\data_classfree',
                             help='directory containing training data')
    parser.add_argument('--save_dir', type=str, default='E:\\wency_ddpm\\result\\IDDPM',
                             help='directory save')
    parser.add_argument('--train_batch_size', type=int, default=8,
                             help='batch size for training')
    parser.add_argument('--train_lr', type=float, default=8e-5,
                             help='learning rate for training')
    parser.add_argument('--train_num_steps', type=int, default=100,
                             help='total number of training steps')
    parser.add_argument('--save_and_sample_every', type=int, default=5)
    parser.add_argument('--gradient_accumulate_every', type=int, default=4,
                             help='number of gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.9995,
                             help='exponential moving average decay')
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--calculate', type=bool, default=False)
    parser.add_argument('--IDDPM-beta_schedule', type=str, default='cosine')
    parser.add_argument('--objective', type=str, default='pred_noise')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = get_arg()
    if opt.model_name == 'DDPM':
        model = Unet(
            dim=opt.dim,
            dim_mults=opt.dim_mults,
            num_classes=opt.num_classes,
            cond_drop_prob=opt.cond_drop_prob)
        diffusion = GaussianDiffusion(
            model,
            image_size=opt.image_size,
            timesteps=opt.timesteps,
        ).cuda()
    else:
        model = Unet_iddpm(
            dim=opt.dim,
            dim_mults=opt.dim_mults,
            learned_variance=True,
        )
        diffusion = LearnedGaussianDiffusion(
            model,
            image_size=opt.image_size,
            timesteps=opt.timesteps,
            beta_schedule=opt.IDDPM_beta_schedule,
            objective=opt.objective,
        ).cuda()

    trainer = Trainer(
        diffusion_model=diffusion,
        folder=opt.data_dir,
        train_batch_size=opt.train_batch_size,
        train_lr=opt.train_lr,
        model_name=opt.model_name,
        train_num_steps=opt.train_num_steps,  # total training steps
        gradient_accumulate_every=opt.gradient_accumulate_every,  # gradient accumulation steps
        ema_decay=opt.ema_decay,  # exponential moving average decay
        amp=opt.amp,  # turn on mixed precision
        calculate_fid=opt.calculate,
        save_and_sample_every=opt.save_and_sample_every,
        results_folder=opt.save_dir,
    )

    trainer.train()