import glob
import csv
# from itertools import izip

csv_name = 'metric.csv'
f = open(csv_name, 'w')
writer = csv.writer(f)

expname = "1103/dtu_{}/viz/test_viz/metric2.txt"
# scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
# scenes=["bicycle", "bonsai", "counter", "garden", "kitchen", "flowers", "treehill"]
# scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
# scenes = ["obj_04_stone", "obj_27_pumpkin2", "obj_29_fabric_toy", "obj_39_potato", "obj_40_pine", "obj_44_fabric_mushroom", "obj_50_fabric_cow", "obj_62-fabric-birthday-cake"]
scenes = ["scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122"]

print(expname)
for scene in scenes:
    file = open(expname.format(scene), 'r')
    for line in file.readlines():
        psnr = float(line.split(' ')[2])
        ssim = float(line.split(' ')[4])
        lpips = float(line.split(' ')[6])
    print(scene, psnr, ssim, lpips)
    writer.writerow([scene, psnr, ssim, lpips])

f.close()


a = zip(*csv.reader(open(csv_name, "r")))
csv.writer(open(csv_name, "w")).writerows(a)
