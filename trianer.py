from tool.base import exists,has_int_squareroot,cycle,divisible_by
from dataset.datasets import Dataset
import math
from pathlib import Path
from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from torch.optim import Adam
from torchvision import utils
from accelerate import Accelerator
from tqdm.auto import tqdm

__version__ = '1.11.0'
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        model_name,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        amp=False,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 2,
        num_samples = 9,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False,
        results_folder,
        max_grad_norm = 1.,
        num_classes = 5,
        save_best_and_latest_only = False
    ):
        super().__init__()
        self.model_name = model_name
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        self.num_classes=num_classes
        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to, model_name=self.model_name)
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process
        # if self.calculate_fid:
        #     from metric.fid_evaluation import FIDEvaluation
        #     if not is_ddim_sampling:
        #         self.accelerator.print(
        #             "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
        #             "Consider using DDIM sampling to save time."
        #         )
        #
        #     self.fid_scorer = FIDEvaluation(
        #         batch_size=self.batch_size,
        #         dl=self.dl,
        #         sampler=self.ema.ema_model,
        #         channels=self.channels,
        #         accelerator=self.accelerator,
        #         stats_dir=results_folder,
        #         device=self.device,
        #         num_fid_samples=num_fid_samples,
        #         inception_block_idx=inception_block_idx
        #     )
        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x,y = next(self.dl)
                    x = x.to(device)
                    y = y.to(device)
                    with self.accelerator.autocast():
                        loss = self.model(x,classes = y)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()#等所有进程运行结束
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)#梯度剪裁
                #优化器调整
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    #是否开启验证保存图像效果
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):#判断是否到验证的步骤
                        self.ema.ema_model.eval()#开始验证
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            image_classes = torch.randint(0, self.num_classes, (9,)).cuda()
                            all_images = self.ema.ema_model.sample(classes = image_classes, cond_scale=6.)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # 是否计算FID
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')