"""
Training code for adversarial patch training
(Option A: Differentiable CAPGen palette/logits parameterization)
"""

import os
import sys
import time
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch import optim  # <<< make sure optim is imported
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans  # >>> Option A: used for palette build (CPU, non-diff)

import PIL
import load_data
from load_data import *
import patch_config
import weather

torch.cuda.set_device(0)  # select gpu to run on

# Reproducibility
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# ======================================================================
# >>> Option A: Differentiable CAPGen (palette + logits → softmax mixing)
# ======================================================================

class CapGenModule(torch.nn.Module):
    """
    Differentiable CAPGen:
      - Fixed palette (K colors) from background(s)
      - Learn per-pixel logits over K colors
      - Patch is a convex mixture of palette colors via softmax(logits/τ)
    """
    def __init__(self, palette_rgb: torch.Tensor, patch_size, tau: float = 0.07):
        """
        Args:
            palette_rgb: [K, 3] float in [0,1] on the same device as the model
            patch_size: (H, W)
            tau: softmax temperature (you can anneal it across epochs)
        """
        super().__init__()
        if palette_rgb.dim() != 2 or palette_rgb.shape[1] != 3:
            raise ValueError("palette_rgb must be [K,3] in [0,1]")
        K = palette_rgb.shape[0]
        H, W = patch_size

        self.tau = float(tau)
        self.register_buffer("palette", palette_rgb)                       # fixed buffer [K,3]
        self.logits = torch.nn.Parameter(torch.randn(1, H, W, K) * 0.01)  # learnable [1,H,W,K]

    def forward(self):
        """
        Returns:
            patch: [1,3,H,W] in [0,1]
            r:     [1,H,W,K] soft assignments (useful for optional regularizers/inspection)
        """
        # Softmax over K colors for each pixel:
        r = F.softmax(self.logits / max(self.tau, 1e-6), dim=-1)  # [1,H,W,K]
        # Weighted sum of palette colors -> RGB:
        patch = torch.einsum('bhwk,kc->bhwc', r, self.palette)    # [1,H,W,3]
        patch = patch.permute(0, 3, 1, 2).contiguous()            # [1,3,H,W]
        return patch.clamp(0, 1), r

    def set_tau(self, tau_new: float):
        self.tau = float(tau_new)


def build_palette_from_tensor(image_tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build a K-color palette from a background image tensor using K-means (CPU; non-differentiable).
    Args:
        image_tensor: [3,H,W] in [0,1], (device doesn't matter; moved to CPU here)
        k: number of colors
    Returns:
        palette: [K,3] float32 in [0,1] (on CPU; move to CUDA in caller)
    """
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError("image_tensor must be [3,H,W]")
    img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy().reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [K,3]
    centers.clamp_(0, 1)
    return centers


# ===========================================
# Existing PatchTrainer with minimal edits
# ===========================================

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

        # >>> Option A: placeholders
        self.capgen_mod = None  # will hold CapGenModule if enabled

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
        """

        # --------------------------
        # Config + sane fallbacks
        # --------------------------
        use_capgen = getattr(self.config, "use_capgen", True)  # default ON for Option A
        capgen_image_path = getattr(self.config, "capgen_image_path", None)
        capgen_num_colors = int(getattr(self.config, "capgen_num_colors", 6))
        capgen_tau_start = float(getattr(self.config, "capgen_tau_start", 0.12))
        capgen_tau_end = float(getattr(self.config, "capgen_tau_end", capgen_tau_start))  # const if equal

        # Network and training parameters
        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = self.config.n_epochs
        max_lab = self.config.max_lab

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # ===========================================
        # Patch initialization
        # ===========================================
        device = torch.device("cuda")

        if use_capgen:
            # >>> Option A: Build palette (K-means) from a representative background
            if capgen_image_path is None:
                raise ValueError("config.capgen_image_path must be set when use_capgen=True")

            bg_image = transforms.ToTensor()(Image.open(capgen_image_path).convert('RGB'))
            palette = build_palette_from_tensor(bg_image, capgen_num_colors).to(device)  # [K,3] on CUDA

            # Create differentiable CAPGen module
            self.capgen_mod = CapGenModule(
                palette_rgb=palette,
                patch_size=self.config.patch_size,
                tau=capgen_tau_start
            ).to(device)

            # For visualizing the *initial* patch:
            with torch.no_grad():
                init_patch, _ = self.capgen_mod()
            adv_patch_preview = init_patch.detach().cpu()

        else:
            # Fallback to original random init (pixel-space parameter)
            adv_patch_cpu = self.generate_patch('one_random') if self.config.patch_num == 1 else \
                            (self.generate_patch('three_random') if self.config.patch_num == 3 else
                             self.generate_patch('two_random'))
            adv_patch_cpu.requires_grad_(True)
            adv_patch_preview = adv_patch_cpu.detach().cpu()

        # Create output folders
        if not os.path.exists(self.folder_selection):
            os.makedirs(self.folder_selection)
            os.makedirs(self.folder_selection + '/PATCH_ITERATIONS')

        # Save initial patch(es)
        for i in range(adv_patch_preview.size(0)):
            im = transforms.ToPILImage('RGB')(adv_patch_preview[i])
            plt.imshow(im)
            plt.savefig(f'{self.folder_selection}/PATCH_ITERATIONS/{time_str}_initial_{i}.jpg')
            plt.close()

        # Dataloader
        train_loader = torch.utils.data.DataLoader(
            LoadDataset(self.config.img_dir,
                        self.config.lab_dir,
                        max_lab,
                        imgsize=self.config.input_size,
                        shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )

        self.epoch_length = len(train_loader)
        print(f'Number of training steps per epoch: {len(train_loader)}\n')

        # Optimizer + scheduler
        if use_capgen:
            optimizer = optim.Adam(self.capgen_mod.parameters(),
                                   lr=self.config.start_learning_rate, amsgrad=True)
        else:
            optimizer = optim.Adam([adv_patch_cpu],
                                   lr=self.config.start_learning_rate, amsgrad=True)

        scheduler = self.config.scheduler_factory(optimizer)

        # Training info
        print("\nTRAINING INFORMATION:")
        print("YOLO input size:", img_size)
        print("Batch size:", batch_size)
        print("Number of epochs:", n_epochs)
        print("Maximum number of labels per image:", max_lab)
        if use_capgen:
            print("CAPGen: ON  (K =", capgen_num_colors, ", tau_start =", capgen_tau_start, ")")
        else:
            print("CAPGen: OFF (random pixel init)")
        print("\n")

        # ===========================================
        # Main training loop
        # ===========================================
        et0 = time.time()
        for epoch in range(n_epochs):
            # Optional τ annealing (linear)
            if use_capgen and capgen_tau_end != capgen_tau_start:
                t = epoch / max(n_epochs - 1, 1)
                tau_now = (1 - t) * capgen_tau_start + t * capgen_tau_end
                self.capgen_mod.set_tau(tau_now)

            ep_nps_loss = 0.0
            ep_tv_loss = 0.0
            ep_det_loss = 0.0
            ep_loss_acc = 0.0

            bt0 = time.time()

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(train_loader),
                desc=f'Running epoch {epoch}',
                total=self.epoch_length
            ):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()

                    # -----------------------------
                    # Get current adversarial patch
                    # -----------------------------
                    if use_capgen:
                        adv_patch, r = self.capgen_mod()   # >>> Option A: generated each iteration
                    else:
                        adv_patch = adv_patch_cpu.cuda()

                    # Apply transformations & paste
                    if self.config.patch_num == 1:
                        adv_batch_t = self.patch_transformations(
                            adv_patch, lab_batch,
                            img_size=self.config.input_size, size=self.config.patch_scale,
                            do_rotate=True, rand_loc=False
                        )
                    elif self.config.patch_num in (2, 3):
                        adv_batch_t = self.patch_transformations(
                            adv_patch, lab_batch,
                            img_size=self.config.input_size, size=self.config.patch_scale,
                            do_rotate=False, rand_loc=False
                        )

                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    # Plot (optional quick save of 1 sample)
                    img = p_img_batch[0, :, :, :]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save(f'{self.folder_selection}/patch_image_full.jpg')

                    # Weather (optional)
                    if self.config.weather_augmentations == 'on':
                        for i in range(p_img_batch.size(0)):
                            weather_type = random.randint(0, 6)
                            if weather_type == 0:
                                p_img_batch[i, :, :, :] = weather.brighten(p_img_batch[i, :, :, :])
                            elif weather_type == 1:
                                p_img_batch[i, :, :, :] = weather.darken(p_img_batch[i, :, :, :])
                            elif weather_type == 2:
                                p_img_batch[i, :, :, :] = weather.add_snow(p_img_batch[i, :, :, :])
                            elif weather_type == 3:
                                p_img_batch[i, :, :, :] = weather.add_rain(p_img_batch[i, :, :, :])
                            elif weather_type == 4:
                                p_img_batch[i, :, :, :] = weather.add_fog(p_img_batch[i, :, :, :])
                            elif weather_type == 5:
                                p_img_batch[i, :, :, :] = weather.add_autumn(p_img_batch[i, :, :, :])
                            elif weather_type == 6:
                                p_img_batch[i, :, :, :] = p_img_batch[i, :, :, :]

                        img = p_img_batch[0, :, :, :]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        img.save(f'{self.folder_selection}/patch_image_full_weather.jpg')

                    # Resize to YOLO input
                    p_img_batch = F.interpolate(
                        p_img_batch, (self.darknet_model.height, self.darknet_model.width)
                    )

                    img = p_img_batch[0, :, :, :]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save(f'{self.folder_selection}/patch_image_resize.jpg')

                    # Forward → loss terms
                    output = self.darknet_model(p_img_batch)

                    max_detection = self.detection_score(output)
                    det_loss = torch.mean(max_detection)
                    nps_loss = self.nps_calculator(adv_patch) * 0.01
                    tv_loss = self.tv_calculator(adv_patch) * 2.5

                    # Optional: entropy regularizer on r (encourage blends early)
                    # If not using CAPGen, set ent_loss=0
                    if use_capgen:
                        r_eps = 1e-8
                        ent = -(r * (r + r_eps).log()).sum(-1).mean()  # mean pixel entropy
                        entropy_weight = float(getattr(self.config, "capgen_entropy_weight", 0.0))
                        ent_loss = -entropy_weight * ent
                    else:
                        ent_loss = torch.tensor(0.0, device=device)

                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1, device=device)) + ent_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss_acc += loss.detach().item()

                    # Backprop + update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Pixel clamp no longer needed (capgen_mod clamps in forward).
                    # If random-pixel path (use_capgen=False), keep clamp:
                    if not use_capgen:
                        adv_patch_cpu.data.clamp_(0, 1)

                    # TB logging
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        if use_capgen:
                            self.writer.add_scalar('capgen/tau', self.capgen_mod.tau, iteration)

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        # Free GPU memory
                        del adv
