# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement

": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.6

Authors: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''
import matplotlib

matplotlib.use('agg')
import os.path
import data
from option import args
from datacode import dataset
from tqdm import tqdm
import torch.utils.data as data
import cv2

import os
import torch
import skimage
from metric import metrics

class Evaluator(object):
    def __init__(self, data_loader, device, args):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader
        self.device = torch.device('cuda')
        self.args = args

    def evaluate(self):
        testLoader = data.DataLoader(dataset=self.data_loader, batch_size=1, shuffle=True,
                                     num_workers=self.args.n_threads, pin_memory=True)

        psnr_avg = 0.0
        ssim_avg = 0.0
        num_batches = 0

        for batch, (lr, hr, x) in enumerate(tqdm(testLoader, ncols=80)):
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            pred = lr.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
            gt = hr.detach().detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
            psnr = skimage.measure.compare_psnr(pred, gt)
            lr = lr.mul(255).byte()
            hr = hr.mul(255).byte()
            img1 = lr.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
            img2 = hr.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            ssim = metrics.calculate_ssim(img1_rgb, img2_rgb)
            psnr_avg = psnr_avg + psnr
            ssim_avg = ssim_avg + ssim
            num_batches += 1

        print('Average PSNR: {}  SSIM: {}  Total image: {}'.format(psnr_avg / num_batches, ssim_avg / num_batches,
                                                                   num_batches))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    global model

    if args.data_name == 'sRGBSID':
        test_input_dir = './result/{}/'.format(args.data_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/sRGBSID/gt/test/'
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'evaluate', '512', args.data_name)

    else:
        test_input_dir = './result/{}/'.format(args.data_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/ELD/{}g/'.format(args.data_name)
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'evaluate', '1024',
                                        args.data_name)

    testing_evaluator = Evaluator(loaderTest, device, args)
    testing_evaluator.evaluate()


if __name__ == '__main__':
    main()
