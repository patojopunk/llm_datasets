"""
Training code for adversarial patch training

"""
import torch
torch.cuda.set_device(0) # select gpu to run on

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
import numpy as np
import random

import weather

# Set random seed for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# CapGen Integration for Du et al.'s PatchTrainer Framework
# Adds functionality to generate environment-adaptive patches using dominant color extraction

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
from PIL import Image

class CapGen:
    def __init__(self, num_colors=5, tau=0.05):
        """
        CapGen: Environment-Adaptive Generator for Adversarial Patches
        
        Args:
            num_colors (int): Number of dominant background colors to extract
            tau (float): Temperature for softmax over color probability distribution
        """
        self.num_colors = num_colors
        self.tau = tau

    def extract_dominant_colors(self, image_tensor):
        """
        Extract dominant colors from the image using k-means clustering.
        
        Args:
            image_tensor (Tensor): Image tensor of shape [3, H, W] with values in [0, 1]
            
        Returns:
            Tensor of shape [num_colors, 3] with RGB color values
        """
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)  # [H*W, 3]
        kmeans = KMeans(n_clusters=self.num_colors, random_state=0).fit(image_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [num_colors, 3]
        return centers

    def generate_patch(self, dominant_colors, patch_size):
        """
        Generate an initial patch based on the dominant colors.
        
        Args:
            dominant_colors (Tensor): [num_colors, 3] RGB
            patch_size (tuple): (H, W)
            
        Returns:
            adv_patch_cpu (Tensor): [1, 3, H, W]
        """
        H, W = patch_size
        log_m = torch.randn(1, H, W, self.num_colors)  # logits for color selection
        r = F.softmax(log_m / self.tau, dim=-1)  # color probabilities per pixel

        # Create patch by weighted sum of color bases
        patch = torch.einsum('bhwc,cd->bhwd', r, dominant_colors)  # [1, H, W, 3]
        patch = patch.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        return patch.clamp(0, 1)

    def from_image(self, image_tensor, patch_size):
        """
        Full pipeline: extract colors from image and generate patch.
        
        Args:
            image_tensor (Tensor): [3, H, W]
            patch_size (tuple): (H, W)

        Returns:
            patch_tensor (Tensor): [1, 3, H, W]
        """
        dominant_colors = self.extract_dominant_colors(image_tensor)
        return self.generate_patch(dominant_colors, patch_size)


# === Integration Point ===
# In train_patch.py (inside PatchTrainer.train), replace:
#     adv_patch_cpu = self.generate_patch('one_random')
# with something like:
#
#     if self.config.use_capgen:
#         bg_image = transforms.ToTensor()(Image.open(self.config.capgen_image_path).convert('RGB'))
#         capgen = CapGen(num_colors=self.config.capgen_num_colors, tau=self.config.capgen_tau)
#         adv_patch_cpu = capgen.from_image(bg_image, self.config.patch_size)
#     else:
#         adv_patch_cpu = self.generate_patch('one_random')
#
# Also extend patch_config to include:
#     - use_capgen (bool)
#     - capgen_image_path (str)
#     - capgen_num_colors (int)
#     - capgen_tau (float)



class PatchTrainer(object):
    def __init__(self, mode, folder):
        
        # mode = 'paper_obj'
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        
        self.patch_transformations = PatchTransformations().cuda()
        self.patch_applier = PatchApplier().cuda()
        
        self.detection_score = MaxDetectionScore(0, 5, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.tv_calculator = TVCalculator().cuda()

        self.writer = self.init_tensorboard(mode)
        
        self.folder_selection = folder

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        self.use_capgen = True
        self.capgen_image_path = 'path/to/sample_background.jpg'
        self.capgen_num_colors = 5
        self.capgen_tau = 0.05

        # Network and training parameters
        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = self.config.n_epochs
        max_lab = self.config.max_lab

        # Start timer
        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate a random patch as a starting point for optimization.
        # if self.config.patch_num == 1:
        #     adv_patch_cpu = self.generate_patch('one_random')   
        # elif self.config.patch_num == 3:
        #     adv_patch_cpu = self.generate_patch('three_random')

        bg_image = transforms.ToTensor()(Image.open(self.config.capgen_image_path).convert('RGB'))
        capgen = CapGen(num_colors=self.config.capgen_num_colors, tau=self.config.capgen_tau)
        adv_patch_cpu = capgen.from_image(bg_image, self.config.patch_size)
     
        # Load existing patch
        # adv_patch_cpu = self.read_image("")
        # adv_patch_cpu = adv_patch_cpu.unsqueeze(0)
        
        # Set up gradient calculation of patch
        adv_patch_cpu.requires_grad_(True)
            
        print("\n")
        print("TRAINING INFORMATION:")
        print("Shape of adversarial patch:", adv_patch_cpu.shape)
        print("YOLO input size:", img_size)
        print("Batch size:", batch_size)
        print("Number of epochs:", n_epochs)
        print("Maximum number of labels per image:", max_lab)
        
        # Create folder to save patches
        if not os.path.exists(self.folder_selection):
            os.makedirs(self.folder_selection)
            os.makedirs(self.folder_selection + '/PATCH_ITERATIONS')       

        for i in range(adv_patch_cpu.size(0)):            
            im = transforms.ToPILImage('RGB')(adv_patch_cpu[i]) ##
            plt.imshow(im) ##
            plt.savefig(f'{self.folder_selection}/PATCH_ITERATIONS/{time_str}_initial_{i}.jpg')

        train_loader = torch.utils.data.DataLoader(LoadDataset(self.config.img_dir, 
                                                                self.config.lab_dir, 
                                                                max_lab, 
                                                                imgsize=self.config.input_size,
                                                                shuffle=True),
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=1)

        self.epoch_length = len(train_loader)
        print(f'Number of training steps per epoch: {len(train_loader)}')
        print("\n")

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_det_loss = 0
            ep_loss = 0
            bt0 = time.time()
            
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                                
                with autograd.detect_anomaly():
                    
                    # training images and its labels
                    img_batch = img_batch.cuda() 
                    lab_batch = lab_batch.cuda() 
                    
                    # # add color jitter to training image
                    # color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                    # img_batch = color_jitter(img_batch)

                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    
                    adv_patch = adv_patch_cpu.cuda() 
                         
                    # apply augmentations on patches
                    if self.config.patch_num == 1:
                        adv_batch_t = self.patch_transformations(adv_patch, lab_batch, img_size=self.config.input_size, size=self.config.patch_scale, do_rotate=True, rand_loc=False)   # ON patch
                    elif self.config.patch_num == 2 or self.config.patch_num == 3:
                        adv_batch_t = self.patch_transformations(adv_patch, lab_batch, img_size=self.config.input_size, size=self.config.patch_scale, do_rotate=False, rand_loc=False)  # OFF patch
                    
                    # apply patches to training images
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t) 
                    
                    # plot
                    img = p_img_batch[0, :, : ,:]  
                    img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()
                    img.save(f'{self.folder_selection}/patch_image_full.jpg')

                    if self.config.weather_augmentations == 'on':
                        # apply weather augmentations                     
                        for i in range(p_img_batch.size(0)):
                        
                            weather_type = random.randint(0,6) 
                            # print('weather:', weather_type)
                        
                            if weather_type == 0:
                                p_img_batch[i, :, : ,:] = weather.brighten(p_img_batch[i, :, : ,:])
                            elif weather_type == 1:
                                p_img_batch[i, :, : ,:] = weather.darken(p_img_batch[i, :, : ,:])
                            elif weather_type == 2:
                                p_img_batch[i, :, : ,:] = weather.add_snow(p_img_batch[i, :, : ,:])
                            elif weather_type == 3:
                                p_img_batch[i, :, : ,:] = weather.add_rain(p_img_batch[i, :, : ,:])
                            elif weather_type == 4:
                                p_img_batch[i, :, : ,:] = weather.add_fog(p_img_batch[i, :, : ,:])
                            elif weather_type == 5:
                                p_img_batch[i, :, : ,:] = weather.add_autumn(p_img_batch[i, :, : ,:])
                            elif weather_type == 6:
                                p_img_batch[i, :, : ,:] = p_img_batch[i, :, : ,:] 
                        
                        # plot
                        img = p_img_batch[0, :, : ,:]  
                        img = transforms.ToPILImage()(img.detach().cpu())
                        # img.show()
                        img.save(f'{self.folder_selection}/patch_image_full_weather.jpg')
                    
                    # resize patched image to 256x256
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width)) 

                    # plot
                    img = p_img_batch[0, :, : ,:]  
                    img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()
                    img.save(f'{self.folder_selection}/patch_image_resize.jpg')
                                       
                    # forward propagate batch of images into model
                    output = self.darknet_model(p_img_batch) 
                    
                    # calculate the 3 terms of the loss function
                    max_detection = self.detection_score(output) 
                    nps = self.nps_calculator(adv_patch)
                    tv = self.tv_calculator(adv_patch) 

                    det_loss = torch.mean(max_detection)
                    nps_loss = nps*0.01 
                    tv_loss = tv*2.5 
                    
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss
                    
                    # back propagation
                    loss.backward()
                    
                    # update patch
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1) # keep patch in image range
        
                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        # self.writer.add_image('patch', adv_patch_cpu, iteration)
                    
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_detection, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            
            # save plot of patch at each epoch 
            for i in range(adv_patch_cpu.size(0)):            
                im = transforms.ToPILImage('RGB')(adv_patch_cpu[i]) ##
                plt.imshow(im) ##
                plt.savefig(f'{self.folder_selection}/PATCH_ITERATIONS/{time_str}_epoch_{epoch}_{i}.jpg')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)

                # save printable patches (.jpg format)
                for i in range(adv_patch_cpu.size(0)):
                    im = transforms.ToPILImage('RGB')(adv_patch_cpu[i])
                    plt.imshow(im) ##
                    # plt.show() ##
                    im.save(f'{self.folder_selection}/patch_{i}.jpg') 

                del adv_batch_t, output, max_detection, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()
            

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray', random' or black/white.
        :return:
        """

        if type == 'one_gray':
            adv_patch_cpu = torch.full((1, 3, self.config.patch_size[0], self.config.patch_size[1]), 0.5)
        elif type == 'one_random':
            adv_patch_cpu = torch.rand((1, 3, self.config.patch_size[0], self.config.patch_size[1]))
        elif type == 'two_random':
            adv_patch_cpu = torch.rand((2, 3, self.config.patch_size[0], self.config.patch_size[1]))
        elif type == 'three_random':
            adv_patch_cpu = torch.rand((3, 3, self.config.patch_size[0], self.config.patch_size[1]))
            
        return adv_patch_cpu


    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size[0], self.config.patch_size[1]))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        
        return adv_patch_cpu


def main():

    # trainer = PatchTrainer('expX', 'experiment00')
    trainer = PatchTrainer(sys.argv[1], sys.argv[2])
    trainer.train()

if __name__ == '__main__':
    main()
