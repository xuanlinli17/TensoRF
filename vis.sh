# scenes="stump room kitchen garden counter bonsai bicycle"
scenes="bonsai"
for scene in $scenes
do
	python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_50k_res512_tv1_2_6_test --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
done


# # scenes="fern flower fortress horns leaves orchids room trex"
# # for scene in $scenes
# # do
# # 	python train.py --config configs/flower.txt --datadir /home/sarahwei/dataset/nerf_llff_data/${scene}/ --expname tensorf_llff_${scene}_v2 --data_dim_color 12 --featureC 64
# # done

# scenes="fern flower fortress horns leaves orchids room trex"
# for scene in $scenes
# do
# 	python train.py --config configs/flower.txt --datadir /home/sarahwei/dataset/nerf_llff_data/${scene}/ --expname test --data_dim_color 27 --shadingMode SH
# done

