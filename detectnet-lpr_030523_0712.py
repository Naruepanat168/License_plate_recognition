#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
import datetime
import math
import os
import time 
#import easyocr
import pytesseract
import numpy as np
from PIL import Image

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage, Log, cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() + jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--snapshots", type=str, default="images/test/detections", help="output directory of detection snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")
parser.add_argument("--hold", type=float, default=0.5, help="hold time in second after captured image was saved" )
parser.add_argument("--firm", type=int, default=5, help="number of loop time with same OCR read" ) 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# make sure the snapshots dir exists
os.makedirs(opt.snapshots, exist_ok=True)

# set OCR reader language
#reader = easyocr.Reader(['th'])

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	timestamp = datetime.datetime.now().strftime(opt.timestamp)

	#for detection in detections:
	#	print(detection)
        
	for idx, detection in enumerate(detections):
		print(detection)
		#time.sleep(opt.hold)
		roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
		snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
		cudaCrop(img, snapshot, roi)
		cudaDeviceSynchronize()
		#ocr_result = reader.readtext(snapshot)
		#tess = Image.fromarray(snapshot.astype(np.uint8))
		#txt = pytesseract.image_to_string(tess, lang='tha')
		if timestamp[14] == "0":
			saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}-p.jpg"), snapshot)
			file_snap = os.path.join(opt.snapshots, f"{timestamp}-{idx}-p.jpg")
			tess = Image.open(file_snap)
			txt = pytesseract.image_to_string(tess, lang='tha')
			f = open(os.path.join(opt.snapshots, f"{timestamp}-{idx}.txt"),"x")
			f.write(txt)
			f.close()
			#saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}-full.jpg"), img)


		del snapshot

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break


