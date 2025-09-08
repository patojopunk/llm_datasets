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
#import weather

# [MLflow] — minimal import and (optional) autolog
import mlflow  # [MLflow]
import mlflow.pytorch  # [MLflow]

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
    print ('got here 4')
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError("image_tensor must be [3,H,W]")
    print ('got here 5')
    img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy().reshape(-1, 3)
    print ('got here 6:', k, img_np.shape)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_np)
    print ('got here 7')
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [K,3]
    print ('got here 8')
    centers.clamp_(0, 1)
    print ('got here 9')
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

    # [MLflow] small helper to log params safely
    def _mlflow_log_config(self):
        try:
            cfg = {}
            for k, v in vars(self.config).items():
                # only simple serializable params
                if isinstance(v, (int, float, str, bool)) or v is None:
                    cfg[k] = v
            if cfg:
                mlflow.log_params(cfg)
        except Exception as e:
            print(f"[MLflow] Skipping config param logging: {e}")

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        """
        print ('got here 0')

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
        save_every = 1

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # ===========================================
        # Patch initialization
        # ===========================================
        device = torch.device("cuda")
        print ('got here 1')

        if use_capgen:
            # >>> Option A: Build palette (K-means) from a representative background
            if capgen_image_path is None:
                raise ValueError("config.capgen_image_path must be set when use_capgen=True")

            print ('got here 2')
            bg_image = transforms.ToTensor()(Image.open(capgen_image_path).convert('RGB'))
            palette = build_palette_from_tensor(bg_image, capgen_num_colors).to(device)  # [K,3] on CUDA
            print ('got here 3')

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
            print ('got here 4')

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

        # [MLflow] — log run-level info and some one-off params/artifacts
        mlflow.set_tag("mode", getattr(self.config, "name", "unknown"))  # lightweight tag
        self._mlflow_log_config()  # params from config
        try:
            mlflow.log_params({
                "use_capgen": use_capgen,
                "capgen_num_colors": capgen_num_colors,
                "capgen_tau_start": capgen_tau_start,
                "capgen_tau_end": capgen_tau_end,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "img_size": img_size
            })
        except Exception as e:
            print(f"[MLflow] Skipping basic param logging: {e}")

        # Optionally log the initial palette and preview
        try:
            if use_capgen:
                # save palette
                npy_path = os.path.join(self.folder_selection, "palette.npy")
                np.save(npy_path, self.capgen_mod.palette.detach().cpu().numpy())
                mlflow.log_artifact(npy_path)
        except Exception as e:
            print(f"[MLflow] Skipping palette artifact: {e}")

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

                    # TB logging + [MLflow] metrics
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        # TensorBoard (existing)
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        if use_capgen:
                            self.writer.add_scalar('capgen/tau', self.capgen_mod.tau, iteration)

                        # [MLflow] mirror key metrics with a step index
                        try:
                            mlflow.log_metrics({
                                "total_loss": float(loss.detach().cpu().numpy()),
                                "det_loss": float(det_loss.detach().cpu().numpy()),
                                "nps_loss": float(nps_loss.detach().cpu().numpy()),
                                "tv_loss": float(tv_loss.detach().cpu().numpy()),
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                **({"capgen_tau": float(self.capgen_mod.tau)} if use_capgen else {})
                            }, step=int(iteration))
                        except Exception as e:
                            print(f"[MLflow] Metric logging skipped: {e}")

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    # else:
                    #     # Free GPU memory
                    #     del adv

            # >>> ADD: save current patch after each iteration
            if (i_batch % save_every) == 0:
                with torch.no_grad():
                    if use_capgen:
                        cur_patch, _ = self.capgen_mod()      # [1,3,H,W]
                    else:
                        # adv_patch_cpu is the trained tensor on CPU
                        cur_patch = adv_patch_cpu.clamp(0, 1) # [N,3,H,W] where N is 1/2/3 depending on patch_num

                    # Save each patch slice (handles 1/2/3 patches)
                    cur_patch_cpu = cur_patch.detach().cpu()
                    for p_idx in range(cur_patch_cpu.size(0)):
                        im = transforms.ToPILImage('RGB')(cur_patch_cpu[p_idx])
                        im.save(f'{self.folder_selection}/PATCH_ITERATIONS/'
                                f'{time_str}_e{epoch:03d}_i{i_batch:05d}_p{p_idx}.jpg')



            # [MLflow] epoch-level aggregates (optional, cheap)
            try:
                mlflow.log_metrics({
                    "epoch_total_loss_mean": ep_loss_acc / max(self.epoch_length, 1),
                    "epoch_det_loss_mean": ep_det_loss / max(self.epoch_length, 1),
                    "epoch_nps_loss_mean": ep_nps_loss / max(self.epoch_length, 1),
                    "epoch_tv_loss_mean": ep_tv_loss / max(self.epoch_length, 1),
                    "epoch_time_sec": time.time() - bt0
                }, step=int((epoch + 1) * self.epoch_length))
            except Exception as e:
                print(f"[MLflow] Epoch aggregate logging skipped: {e}")

        # [MLflow] — log artifacts directory and model snapshot
        try:
            # log the whole output folder (images, previews)
            if os.path.isdir(self.folder_selection):
                mlflow.log_artifacts(self.folder_selection, artifact_path="outputs")

            # Save final patch tensor and/or model
            if use_capgen and self.capgen_mod is not None:
                # Save the CAPGen module
                mlflow.pytorch.log_model(self.capgen_mod, artifact_path="capgen_mod")
                # Also save final generated patch
                with torch.no_grad():
                    final_patch, _ = self.capgen_mod()
                fp_path = os.path.join(self.folder_selection, "final_patch.pt")
                torch.save(final_patch.detach().cpu(), fp_path)
                mlflow.log_artifact(fp_path, artifact_path="outputs")
            else:
                fp_path = os.path.join(self.folder_selection, "final_patch.pt")
                torch.save(adv_patch_cpu.detach().cpu(), fp_path)
                mlflow.log_artifact(fp_path, artifact_path="outputs")
        except Exception as e:
            print(f"[MLflow] Final artifact/model logging skipped: {e}")

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

# Keep your argv override
sys.argv = ['train_patch_capgen.py', 'carpark', 'cg006']

def main():

    # [MLflow] — light-touch run wrapper with names from your args
    try:
        mlflow.set_experiment(f"advpatch_{sys.argv[1]}")  # groups runs by mode
    except Exception as e:
        print(f"[MLflow] set_experiment skipped: {e}")

    # Optional: enable generic autolog (won’t interfere with manual logging)
    try:
        mlflow.pytorch.autolog(log_models=False)  # keep manual model logging; metrics still fine
    except Exception as e:
        print(f"[MLflow] autolog skipped: {e}")

    with mlflow.start_run(run_name=sys.argv[2]):  # run name = folder/label you already pass
        # trainer = PatchTrainer('expX', 'experiment00')
        trainer = PatchTrainer(sys.argv[1], sys.argv[2])
        # Record a couple of basic tags early
        try:
            mlflow.set_tag("script", "train_patch_capgen.py")
            mlflow.set_tag("patch_num", getattr(trainer.config, "patch_num", "unknown"))
        except Exception:
            pass
        trainer.train()

if __name__ == '__main__':
    main()
