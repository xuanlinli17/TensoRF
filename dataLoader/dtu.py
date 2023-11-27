import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from sklearn.cluster import KMeans

from .ray_utils import *

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class DtuDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        
        self.downsample = 2
        self.split = split

        cams = np.load(os.path.join(self.root_dir, "cameras_sphere.npz"))

        img_sample = cv2.imread(os.path.join(self.root_dir, 'image', '000000.png'))
        H, W = img_sample.shape[0], img_sample.shape[1]

        w, h = int(W / self.downsample + 0.5), int(H / self.downsample + 0.5)

        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.factor = w / W

        mask_dir = os.path.join(self.root_dir, 'mask')
        self.has_mask = True
        self.apply_mask = True
        
        self.directions = []
        self.poses, self.imgs, self.intrinsic = [], [], []

        n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

        for i in range(n_images):
            world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
            
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w[:3,:4])    

            self.intrinsic.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))     

            if self.split in ['train', 'test']:
                img_path = os.path.join(self.root_dir, 'image', f'{i:06d}.png')
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]

                mask_path = os.path.join(mask_dir, f'{i:03d}.png')
                mask = Image.open(mask_path).convert('L') # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.LANCZOS)
                mask = TF.to_tensor(mask)[0]

                img = img * mask[..., None] + (1 - mask[..., None])
                self.imgs.append(img)

        self.poses = torch.stack(self.poses, dim=0)
        # concat 0,0,0,1
        self.poses = torch.cat([self.poses, torch.tensor([0, 0, 0, 1.], dtype=torch.float32).repeat(self.poses.shape[0], 1, 1)], axis=-2)

        self.imgs = torch.stack(self.imgs, dim=0)
        self.intrinsic = torch.stack(self.intrinsic, dim=0)     # focal_length, focal_length, x, y

        # n_test_traj_steps = 60
        # self.render_path = self.poses # get_spiral(self.poses[:,:3,:], N_views=n_test_traj_steps)
        # self.render_intrinsic = self.intrinsic #self.intrinsic[[0]].repeat(n_test_traj_steps, 1)


        self.read_meta()
        self.define_proj_mat()

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        w, h = self.img_wh

        self.all_rays = []
        self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.imgs) // self.N_vis
        test_idx = kmeans_downsample(self.poses[:,:3,3].numpy(), 5)
        if self.split == "train":
            # idxs = list(range(0, len(self.imgs), img_eval_interval))
            # idxs = list(range(0, 6))
            idxs = list(set(range(len(self.imgs))) - set(test_idx))
        else:
            # idxs = list(range(0, len(self.imgs), img_eval_interval))
            idxs = test_idx

        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#


            # ray directions for all pixels, same for all images (same H, W, focal)
            self.directions = get_ray_directions(h, w, [self.intrinsic[i][0],self.intrinsic[i][1]])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            self.intrinsics = torch.tensor([[self.intrinsic[i][0],0,w/2],[0,self.intrinsic[i][1],h/2],[0,0,1]]).float()

            c2w = self.poses[i]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = self.imgs[idxs].view(-1, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.imgs[idxs]
        self.poses = self.poses[idxs]

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
