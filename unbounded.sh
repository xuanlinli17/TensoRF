scenes="stump room kitchen garden counter bonsai bicycle"
for scene in $scenes
do
	python train.py --config configs/unbounded.txt --datadir /home/sarahwei/dataset/360_v2/${scene}/ --expname tensorf_360_${scene}_unbounded_50k_res512_tv1_2_6_test --data_dim_color 12 --featureC 64 --TV_weight_density 1.0 --TV_weight_app 1.0
done