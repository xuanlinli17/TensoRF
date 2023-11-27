scenes="stump room kitchen garden counter bonsai bicycle flowers treehill"
# scenes="bonsai"
for scene in $scenes
do
	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei
	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0 --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0_bg2 --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0 
	python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --ckpt log/tensorf_360_${scene}_unbounded_anpei_my_0_0_bg2/tensorf_360_${scene}_unbounded_anpei_my_0_0_bg2.th --render_test 1 --render_only 1

	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname test --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
done

# python train.py --config configs/unbounded.txt --datadir /xiwei_fast_cephfs/dataset/360_v2/flowers/ --expname tensorf_360_flowers_unbounded_anpei_my_0_0_app27_3x128 --data_dim_color 27 --featureC 128 --TV_weight_density 1.0 --TV_weight_app 1.0

python train.py --config configs/dtu.txt --datadir /home/sarahwei/dataset/data_DTU/dtu_scan24 --ckpt log/tensorf_scan24/tensorf_scan24.th --render_test 1 --render_only 1
