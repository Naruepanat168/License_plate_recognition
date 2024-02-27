
# LPR_Notify token = 9rijSewvMnyK3np8jrxElsKCG1xCuUGS6ba59M4vGTH
# LPR token = oQLDVrUPPsKflDUxU1upcvC5GhvSlR9tZltkUOpl3d6

import jetson.inference
import jetson.utils

import argparse
import sys
import datetime
import math
import os
import re
#import easyocr
import pytesseract
import numpy as np
from PIL import Image
import requests

from jetson_inference import detectNet
from jetson_utils import (cudaImage, cudaToNumpy , videoSource, videoOutput, saveImage, Log, cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)

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
parser.add_argument("--ocr_line", type=int, default=3, help="number of loop time with same OCR read" ) 
parser.add_argument("--line_token", type=str, default="9rijSewvMnyK3np8jrxElsKCG1xCuUGS6ba59M4vGTH", help="LINE Notify token" ) 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# make sure the snapshots dir exists
os.makedirs(opt.snapshots, exist_ok=True)

# Setup Line Notify Token
url = 'https://notify-api.line.me/api/notify'
token = opt.line_token # Line Notify Token

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

        
	for idx, detection in enumerate(detections):
		print(detection)
		
		roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
		snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
		cudaCrop(img, snapshot, roi)
		cudaDeviceSynchronize()
		
		image_array = jetson.utils.cudaToNumpy(snapshot)
		tess = Image.fromarray(image_array, 'RGB')

		try:
			txt = pytesseract.image_to_string(tess, lang='tha')
		except:
			print("Tesseract Not Active")
			txt = ''
			
			
		words = txt.split("\n")

		if txt != '' and len(words) >= opt.ocr_line:
				
			saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}.jpg"), snapshot)
			img_file = {'imageFile': open(os.path.join(opt.snapshots, f"{timestamp}-{idx}.jpg"),'rb')} #Load LPR picture File
			
			saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}-full.jpg"), img)
			img_full = {'imageFile': open(os.path.join(opt.snapshots, f"{timestamp}-{idx}-full.jpg"),'rb')} #Load Full picture File
			
			dt = timestamp[6]+timestamp[7]+'/'+timestamp[4]+timestamp[5]+'/'+timestamp[0]+timestamp[1]+timestamp[2]+timestamp[3]+' '+timestamp[9]+timestamp[10]+':'+timestamp[11]+timestamp[12]+':'+timestamp[13]+timestamp[14]
			data = {'message': ("แจ้งเตือนยานพาหนะ Cam1 : "+'\n'+dt)}
			headers = {'Authorization':'Bearer ' + token}
			session = requests.Session()
			session_post = session.post(url, headers=headers, files=img_full, data =data)
		
			data = {'message': (txt)}
			session = requests.Session()
			session_post = session.post(url, headers=headers, files=img_file, data =data)


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


