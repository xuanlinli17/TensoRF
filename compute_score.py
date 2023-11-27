import glob
import csv
# from itertools import izip

csv_name = 'metric.csv'
f = open(csv_name, 'w')
writer = csv.writer(f)

expname = "log/tensorf_{}_VM_kmeans_16v/imgs_test_all/mean.txt"
scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
# scenes=["bicycle", "bonsai", "counter", "garden", "kitchen", "flowers", "treehill"]
# scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

print(expname)
for scene in scenes:
    file = open(expname.format(scene), 'r')
    lines = file.readlines()
    psnr = float(lines[0].strip())
    ssim = float(lines[1].strip())
    lpips = float(lines[3].strip())
    print(scene, psnr, ssim, lpips)
    writer.writerow([scene, psnr, ssim, lpips])

f.close()


a = zip(*csv.reader(open(csv_name, "r")))
csv.writer(open(csv_name, "w")).writerows(a)
