import os
import time
import datetime
import torch
import torch.nn as nn
import glob
import os.path as osp

from model import Unet
from torchvision.utils import save_image


class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size    = config['MODEL_CONFIG']['IMG_SIZE']
        assert self.img_size in [256, 512]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.lr          = float(config['TRAINING_CONFIG']['LR'])
        self.lambda_cls = config['TRAINING_CONFIG']['LAMBDA_CLS']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.loss = nn.L1Loss()

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):
        self.model = Unet(n_channels=3, n_classes=3).to(self.gpu)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, (self.beta1, self.beta2))
        self.print_network(self.model, 'Model')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-model.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x.split('-')[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        model_path = os.path.join(self.model_dir, '{}-model.ckpt'.format(epoch))
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(self.gpu)
        return epoch

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        fixed_image, fixed_target = next(data_iter)

        fixed_image = fixed_image.to(self.gpu)
        fixed_target = fixed_target.to(self.gpu)

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    images, target = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    images, target = next(data_iter)

                images = images.to(self.gpu)
                target = target.to(self.gpu)

                loss_dict = dict()

                pred_edge = self.model(images)
                loss = self.loss(pred_edge, target)

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict['loss'] = loss.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report = list()
                    image_report.append(fixed_image)
                    image_report.append(fixed_target)
                    image_report.append(self.model(fixed_image))
                    x_concat = torch.cat(image_report, dim=3)
                    sample_path = osp.join(self.sample_dir, '{}-images.jpg'.format(str(e + 1).zfill(len(str(self.epoch)))))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                    print('Saved real and fake images into {}...'.format(self.sample_dir))
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                model_path = os.path.join(self.model_dir, '{}-model.ckpt'.format(e + 1))
                torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self):
        pass

