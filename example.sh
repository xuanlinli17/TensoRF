scenes="chair drums ficus lego hotdog mic materials ship"
for scene in $scenes
do
        python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/${scene} --expname tensorf_${scene}_VM_2_6 --data_dim_color 12 --featureC 64 --view_pe 6 --fea_pe 2
done