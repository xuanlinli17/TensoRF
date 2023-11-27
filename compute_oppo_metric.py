import glob
import numpy as np
import os
import cv2
import imageio
import cmapy
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm
import json

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name='vgg', device='cuda'):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device).float()
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device).float()
    return __LPIPS__[net_name](gt, im, normalize=True).item()

# fix
FLOAT = False

BLACK = False
CROP_OURS = False
CROP_TENSORF = False

scenes = ["obj_04_stone", "obj_27_pumpkin2", "obj_29_fabric_toy", "obj_39_potato", "obj_40_pine", "obj_44_fabric_mushroom", "obj_50_fabric_cow", "obj_62-fabric-birthday-cake"]
# scenes = ["obj_04_stone"]
camera_file = os.path.join("/home/sarahwei/dataset/openillumination/lighting_patterns/transforms_alignz_test.json")
with open(camera_file, 'r') as f:
    meta = json.load(f)['frames']
downsample = 4

expnames = [
            "/home/sarahwei/code/FreeNeRF/out/oppo/{}4/test/"
            ]
for expname in expnames:
    for scene in scenes:
        root_dir = "/home/sarahwei/dataset/openillumination/lighting_patterns/{}".format(scene)
        ours_filename = os.path.join(expname.format(scene), "{}.png")
        save_filename = os.path.join(expname.format(scene), "metric2.txt")
        gt_filename = os.path.join("/home/sarahwei/dataset/nerf_synthetic/{}/test/".format(scene), "r_{}.png")
        print(gt_filename)

        txt_file = open(save_filename, "w")

        ours_psnrs = []
        ours_ssims = []
        ours_lpipss = []
        tensorf_psnrs = []
        tensorf_ssims = []
        tensorf_lpipss = []
        print(save_filename)
        
        i = 0
        for k, v in tqdm(meta.items()):
            imgid = v['file_path'].split('/')[-1]

            image_path = os.path.join(root_dir, f"Lights/013/raw_undistorted/{imgid}.JPG")
            # img = Image.open(image_path)
            _gt = cv2.imread(image_path) / 255.
            mask_path = os.path.join(root_dir, f"output/com_masks/{imgid}.png")
            mask = cv2.imread(mask_path, 2) > 0
            _gt = _gt * mask[...,None] + (1 - mask[...,None])

            # downsample _gt by 4
            gt = cv2.resize(_gt, (int(_gt.shape[1]/downsample), int(_gt.shape[0]/downsample)))
            ours = cv2.imread(ours_filename.format(i)) / 255.

            if not FLOAT:
                gt = (gt*255.).astype("uint8")
                ours = (ours*255.).astype("uint8")
                R = 255
            else:
                gt = np.clip(gt, a_min=0., a_max=1.)
                ours = np.clip(ours, a_min=0., a_max=1.)
                R = 1
            
            ours_psnr = cv2.PSNR(ours, gt, R=R)
            ours_ssim = structural_similarity(ours, gt, channel_axis=2, data_range=R)
            ours_lpips = rgb_lpips(gt/255., ours/255.)

            # print(i, ours_psnr, ours_ssim, ours_lpips, '----', tensorf_psnr, tensorf_ssim, tensorf_lpips)
            
            ours_psnrs.append(ours_psnr)
            ours_ssims.append(ours_ssim)
            ours_lpipss.append(ours_lpips)

            i += 1

        print("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f} LPIPS: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims), np.mean(ours_lpips)))

        txt_file.write("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f} LPIPS: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims), np.mean(ours_lpips)))
        txt_file.close()
        # print("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims)))

        # txt_file.write("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims)))
        # txt_file.close()



