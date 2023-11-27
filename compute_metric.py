import glob
import numpy as np
import os
import cv2
import imageio
import cmapy
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

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

scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
# scenes = ["lego"]

expnames = [
            # {:03d}
            # "log/tensorf_{}_VM_kmeans_2v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_3v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_4v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_6v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_8v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_12v/imgs_test_all",
            # "log/tensorf_{}_VM_kmeans_16v/imgs_test_all",
            # "log/tensorf_{}_VM_tv_kmeans_6v/imgs_test_all",
            # "/home/sarahwei/code/factor-fields/logs/{}/imgs_test_all",
            # "/home/sarahwei/code/factor-fields/logs/{}_tv/imgs_test_all",
            # {}
            # "/home/sarahwei/code/FlipNeRF/out/flipnerf_blender4_{}/test/",
            # "/home/sarahwei/code/FlipNeRF/out/flipnerf_blender6_{}/test/",
            "/home/sarahwei/code/FreeNeRF/out/nerf_synthetic/{}4/test/"
            ]
for expname in expnames:
    for scene in scenes:
        # ours_filename = os.path.join(expname.format(scene), "{:03d}.png")
        ours_filename = os.path.join(expname.format(scene), "{}.png")
        save_filename = os.path.join(expname.format(scene), "metric2.txt")
        gt_filename = os.path.join("/home/sarahwei/dataset/nerf_synthetic/{}/test/".format(scene), "r_{}.png")
        print(gt_filename)

        txt_file = open(save_filename, "w")
        if 'test' in gt_filename:
            N_img = 200
        else:
            N_img = len(glob.glob(gt_filename.replace("{:06d}", "{}").format('*')))

        ours_psnrs = []
        ours_ssims = []
        ours_lpipss = []
        tensorf_psnrs = []
        tensorf_ssims = []
        tensorf_lpipss = []
        print(save_filename)
        for i in tqdm(range(N_img)):
            _gt = cv2.imread(gt_filename.format(i),flags=cv2.IMREAD_UNCHANGED) / 255.
            ours = cv2.imread(ours_filename.format(i)) / 255.

            # blend gt
            if BLACK:
                gt = _gt[...,:3] * _gt[...,-1:]
            else:
                gt = (_gt[...,:3] * _gt[...,-1:] + (1 - _gt[...,-1:]))

            # crop pred
            alpha = _gt[...,:-1]
            alpha[alpha != 0] = 1.
            if CROP_OURS:
                if BLACK:
                    ours = ours*alpha
                else:
                    ours = ours*alpha+1 - alpha

            # gt = cv2.imread(gt_filename.format(i)) / 255.     # open for unbounded scenes
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

        print("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f} LPIPS: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims), np.mean(ours_lpips)))

        txt_file.write("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f} LPIPS: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims), np.mean(ours_lpips)))
        txt_file.close()
        # print("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims)))

        # txt_file.write("SCENE-{}: PSNR: {:.2f} SSIM: {:.4f}".format(scene, np.mean(ours_psnrs), np.mean(ours_ssims)))
        # txt_file.close()



