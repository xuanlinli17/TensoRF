# # scenes="stump room kitchen garden counter bonsai bicycle flowers treehill"
# # # scenes="flowers treehill"
# # for scene in $scenes
# # do
# # 	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei
# # 	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0 --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
# # 	python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0_SH --data_dim_color 27 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0 --shadingMode SH
# # done


# scenes="fern flower fortress horns leaves orchids room trex"
# for scene in $scenes
# do
# 	# python train.py --config configs/flower.txt --datadir /home/sarahwei/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2 --data_dim_color 12 --featureC 64
# 	python train.py --config configs/flower.txt --datadir /home/sarahwei/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2_SH --data_dim_color 27 --shadingMode SH
# done

# # scenes="fern flower fortress horns leaves orchids room trex"
# # for scene in $scenes
# # do
# # 	python train.py --config configs/flower.txt --datadir /home/sarahwei/dataset/nerf_llff_data/${scene}/ --expname test --data_dim_color 27 --shadingMode SH
# # done

# python train.py --config configs/lego.txt --datadir /home/sarahwei/dataset/nerf_synthetic/chair --expname tensorf_chair_6view_256_tv_s512 --n_iters 10000 --vis_every 2000 --TV_weight_density 1.0 --N_voxel_final 16777216 --nSamples 512

# scenes="chair drums ficus lego hotdog mic materials ship"
# # scenes="chair"
# for scene in $scenes
# do
#     # views="2 4 8 10 12 14 16"
#     views="6"
#     for view in $views
#     do
#         python train.py --config configs/lego_vm.txt --datadir /home/sarahwei/dataset/nerf_synthetic/${scene} --ckpt log/tensorf_${scene}_VM_kmeans_${view}v/tensorf_${scene}_VM_kmeans_${view}v.th --render_test 1 --render_only 1
#     done
# done


scenes="chair drums ficus lego hotdog mic materials ship"
# scenes="chair"
for scene in $scenes
do
    # views="2 4 8 10 12 14 16"
    views="3"
    for view in $views
    do
        python train.py --config configs/lego_vm.txt --datadir /home/sarahwei/dataset/nerf_synthetic/${scene} --expname tensorf_${scene}_VM_kmeans_${view}v --n_views ${view}
    done
done