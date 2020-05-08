import argparse, time, os
import imageio
import numpy as np
import math
import options.options as option
from utils import util
from solvers import create_solver_db  as create_solver
from data import create_dataloader
from data import create_dataset
import math
import cv2
from dem2ply import dem2ply
from timeit import default_timer as timer
import torch

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    pick_factor = np.max(img1) - np.min(img1)
    # print(pick_factor)
    mse = np.mean((img1 - img2)**2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')

    return 20 * math.log10((pick_factor) / rmse), rmse


parser = argparse.ArgumentParser(description='Test Super Resolution Models')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt)
opt = option.dict_to_nonedict(opt)
TILE_SIZE = 200
MAX_N = 1
# initial configure
scale = opt['scale']
degrad = opt['degradation']
network_opt = opt['networks']
model_name = network_opt['which_model'].upper()
if opt['self_ensemble']: model_name += 'plus'

# create solver (and load model)
solver = create_solver(opt)
# initial configure

region = 'durrenstein'
# region = 'montemagro'
# region = 'forcanada'
# region = 'bassiero'

INPUT_DIR = '/home/ashj/FB_TEST/'+region

overlap = 50
solver = create_solver(opt)

hLR = np.loadtxt(INPUT_DIR + '_15m.dem', delimiter = ',', dtype=np.float32)
hHR = np.loadtxt(INPUT_DIR + '_2m.dem', delimiter = ',', dtype=np.float32)
np.save('hLR', hLR)
np.save('hHR', hHR)
factor = TILE_SIZE - overlap
hshape = hLR.shape
ntiles = (math.ceil(hshape[0]/factor), math.ceil(hshape[1]/factor))
print(ntiles)
extshape = (ntiles[0]* factor + overlap, ntiles[1]* factor + overlap)
hLR = cv2.copyMakeBorder(hLR, 0, extshape[0]-hshape[0], 0, extshape[1]-hshape[1], cv2.BORDER_REPLICATE)
hHR = cv2.copyMakeBorder(hHR, 0, extshape[0]-hshape[0], 0, extshape[1]-hshape[1], cv2.BORDER_REPLICATE)
np.save('hLR', hLR)
np.save('hHR', hHR)
# exit()
print(hshape)
# shape for INPUT_DIR (data blob is N x C x H x W), set data
Ntotal = ntiles[0]*ntiles[1]
N = min(Ntotal, MAX_N)

print(N)
# space for output collection
hout = np.zeros(extshape)
need_HR = False
# iterate over batches
numBatches = math.ceil(Ntotal/N)
print('Num batches ', numBatches)
 
PSNR = 0.0
RMSE = 0.0


mask = np.zeros(extshape)
data = {}
for ti in range(ntiles[0]):
	for tj in range(ntiles[1]):

		tistart = ti * 150 
		tiend =  ti * 150 + 200
		tjstart = tj * 150
		tjend = tj * 150 + 200
		if ti == 0:
			tistart = 0
			tiend = 200
		if tj == 0:
			tjstart = 0
			tjend = 200
		print(tistart, tiend, tjstart, tjend)
		hpart = hLR[tistart:tiend, tjstart:tjend]
		np.save('LR_{}_{}'.format(ti, tj), hpart)
		GT = hHR[tistart:tiend, tjstart:tjend]
		np.save('GT_{}_{}'.format(ti, tj), GT)
		mask[tistart:tiend, tjstart:tjend] += 1
		
		hmean = np.mean(hpart)

		inData = hpart - hmean

		inData = inData[np.newaxis, ...] # create C dimension (C = 1)

		# print('inData shape', inData.shape)
		# print('opart shape', opart.shape)
		inData = torch.from_numpy(inData)
		inData = inData.view([1,1,TILE_SIZE,TILE_SIZE])

		# print('mean', inData.mean())
		# print('inData shape', inData.shape)

		inData = inData.cuda()
		data['LR'] = inData
		data['LR_path'] = 'temp'
		solver.feed_data(data, need_HR=False)

		# calculate forward time
		t0 = time.time()

		solver.test()

		t1 = time.time()
		# total_time.append((t1 - t0))
		tend = timer()
		visuals = solver.get_current_visual2(need_HR=False)
		# sr_list.append(visuals['SR'])

		# calculate PSNR/SSIM metrics on Python
		# if need_HR:
		#     psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
		#     total_psnr.append(psnr)
		#     total_ssim.append(ssim)
		#     path_list.append(os.path.basename(batch['HR_path'][0]).replace('HR', model_name))
		#     print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
		#                                                                            os.path.basename(batch['LR_path'][0]),
		#                                                                            psnr, ssim,
		#                                                                            (t1 - t0)))
		# else:
		#     path_list.append(os.path.basename(batch['LR_path'][0]))
		#     print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
		#                                                os.path.basename(batch['LR_path'][0]),
		#                                                (t1 - t0)))

		# out = model(opart, inData)
		# out = out.detach().cpu().float().numpy()
		# print('Out shape', out.shape)
		

		# hpart = out[0] # remove also C dimension 
		hpart = visuals['SR_3'][0][0]

		hpart = hpart + hmean
		# print(hout[tistart:tiend, tjstart:tjend])
		hout[tistart:tiend, tjstart:tjend] += hpart[...]
		# print(hout[tistart:tiend, tjstart:tjend])

		# exit()
		for idx in range(4):
			out_tile = visuals['SR_{}'.format(idx)][0][0] + hmean
			# print('max at', np.max(np.array(out_tile)))
			# print(out_tile)
			# exit()
			np.save('vis_{}_{}_{}'.format(ti, tj, idx), out_tile)       
		# exit()
# np.save('mask', mask)
# exit()
# crop out the added padding
print(hout.shape)
np.save('padded_hout', hout)
hout = hout / mask
hout = hout[0:hshape[0], 0:hshape[1]]
hLR = hLR[0:hshape[0], 0:hshape[1]]
hHR = hHR[0:hshape[0], 0:hshape[1]]
print('hout shape', hout.shape)

print('Region is : ', region)
psnr, rmse = calculate_psnr(hHR, hout)
print('For output psnr: {}, rmse: {}'.format(psnr, rmse))

psnr, rmse = calculate_psnr(hHR, hLR)
print('For input psnr: {}, rmse: {}'.format(psnr, rmse))

# save output
np.save(INPUT_DIR + '_out', hout)
#dem2ply(hout, INPUT_DIR + '_out', 2)
