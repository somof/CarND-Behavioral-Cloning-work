import os
import csv
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('-i', '--csvfile', type=str, help='input csv file')
parser.add_argument('-d', '--imgdir', type=str, help='input data dir')
args = parser.parse_args()
# parser.print_help()
# print(args)

print('csvfile  : ', args.csvfile)
print('imagedir : ', args.imgdir)

# center,left,right,steering,throttle,brake,speed
samples = []
with open(args.csvfile) as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  
    for line in reader:
        samples.append(line)


# 0-7000
for i, line in enumerate(samples):
    if 0 == i % 700 and i < 7001:
        im0 = Image.open(os.path.join(args.imgdir, line[1].strip()))
        im1 = Image.open(os.path.join(args.imgdir, line[0].strip()))
        im2 = Image.open(os.path.join(args.imgdir, line[2].strip()))

        canvas = Image.new("RGB",(360*3,160),(255,255,255))
        canvas.paste(im0, (0,0))
        canvas.paste(im1, (360,0))
        canvas.paste(im2, (720,0))
        canvas.save('fig/scene_{:05d}.jpg'.format(i), 'JPEG', quality=90, optimize=True)

        canvas = Image.new("RGB",(360*3,70),(255,255,255))
        canvas.paste(im0.crop((0, 70, 360, 140)), (0,0))
        canvas.paste(im1.crop((0, 70, 360, 140)), (360,0))
        canvas.paste(im2.crop((0, 70, 360, 140)), (720,0))
        canvas.save('fig/crop_{:05d}.jpg'.format(i), 'JPEG', quality=90, optimize=True)
