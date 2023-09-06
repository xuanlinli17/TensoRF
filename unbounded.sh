scenes="stump room kitchen garden counter bonsai bicycle flowers treehill"
# scenes="flowers treehill"
for scene in $scenes
do
	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei
	# python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0 --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
	python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_anpei_my_0_0_app27_3x128 --data_dim_color 27 --featureC 128 --TV_weight_density 1.0 --TV_weight_app 1.0
done