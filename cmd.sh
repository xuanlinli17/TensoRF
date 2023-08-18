# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM


# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname test --ckpt log/nautilus/tensorf_chair_VM_t12_6_2_mlp3x16/tensorf_chair_VM_t12_6_2_mlp3x16.th --render_test 1 --render_only 1

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x128 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 128

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64

python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12 --featureC 64

python train.py --config configs/realdata.txt --datadir /home/sarahwei/dataset/realdata --expname tensorf_stone_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 # --export_mesh 1

python train.py --config configs/realdata.txt --ckpt log/tensorf_materials_VM_t12_6_2_mlp3x64/tensorf_materials_VM_t12_6_2_mlp3x64.th --export_mesh 1 

python train.py --config configs/realdata.txt --datadir /home/sarahwei/dataset/realdata --expname tensorf_stone_VM_t27_6_2_app48_SH --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp12x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --mlp_layers 12 --featureC 64 --shadingMode MLP_nFea

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp12x32 --view_pe 6 --fea_pe 2 --data_dim_color 12 --mlp_layers 12 --featureC 32 --shadingMode MLP_nFea

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp12x16 --view_pe 6 --fea_pe 2 --data_dim_color 12 --mlp_layers 12 --featureC 16 --shadingMode MLP_nFea

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp12x8 --view_pe 6 --fea_pe 2 --data_dim_color 12 --mlp_layers 12 --featureC 8 --shadingMode MLP_nFea

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12
# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12

scenes="chair drums ficus lego hotdog mic materials ship"
# scenes="lego"
for scene in $scenes
do
    python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/${scene} --expname tensorf_${scene}_VM_t27_6_2_app48_SH --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH
done


#####------------------------------------- nautilus ------------------------------------------------------------------------

# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x64 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64"

# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x64_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 64 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12"


# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/lego --expname tensorf_lego_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t27_6_2_app48_SH_iters --view_pe 6 --fea_pe 2 --data_dim_color 27 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH"


# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x32_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24"

# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x32_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 32 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12"

# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x16_app24 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 24 --n_lamb_sh 24 --n_lamb_sh 24"


# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/chair --expname tensorf_chair_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/drums --expname tensorf_drums_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ficus --expname tensorf_ficus_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/hotdog --expname tensorf_hotdog_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/materials --expname tensorf_materials_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/mic --expname tensorf_mic_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12;
# python train.py --config configs/lego.txt --datadir /xiwei_fast_cephfs/dataset/nerf_synthetic/ship --expname tensorf_ship_VM_t12_6_2_mlp3x16_app12 --view_pe 6 --fea_pe 2 --data_dim_color 12 --featureC 16 --n_lamb_sh 12 --n_lamb_sh 12 --n_lamb_sh 12"

# nautilus LLFF
scenes="fern flower fortress horns leaves orchids room trex"
for scene in $scenes
do
	python train.py --config configs/flower.txt --datadir /xiwei_fast_cephfs/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2_app48_3x64 --data_dim_color 12 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --featureC 32
done

scenes="fern flower fortress horns leaves orchids room trex"
for scene in $scenes
do
	python train.py --config configs/flower.txt --datadir /xiwei_fast_cephfs/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2_app48_SH --data_dim_color 27 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH
done

python train.py --config configs/flower.txt --datadir /xiwei_fast_cephfs/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2_app48_3x64 --data_dim_color 12 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --featureC 32
python train.py --config configs/flower.txt --datadir /xiwei_fast_cephfs/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2_app48_SH --data_dim_color 27 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sigma 16 --n_lamb_sh 48 --n_lamb_sh 48 --n_lamb_sh 48 --shadingMode SH


# xyz = torch.tensor([-1,-1,-1]).view(-1,3).float().cuda()
# xyz = torch.tensor([1,1,1]).view(-1,3).float().cuda()
# xyz = torch.tensor([0.44,-0.32,-0.11]).view(-1,3).float().cuda()

# coordinate_plane = torch.stack((xyz[..., self.matMode[0]], xyz[..., self.matMode[1]], xyz[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
# coordinate_line = torch.stack((xyz[..., self.vecMode[0]], xyz[..., self.vecMode[1]], xyz[..., self.vecMode[2]]))
# coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2).view(3, -1, 1, 2)#.detach().view(3, -1, 1, 2)

# F.grid_sample(self.app_plane[0], coordinate_plane[[0]], align_corners=False)
# F.grid_sample(self.app_line[0], coordinate_line[[0]], align_corners=False)