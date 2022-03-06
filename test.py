# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}

Tested with Pytorch 1.7.1, Python 3.6

Author: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''

import data
from option import args
from datacode import dataset
import torchvision.utils as vutils
import os
import torch
from tqdm import tqdm
import torch.utils.data as data
from models import network


def test(model, args, loader_test, device):
    torch.set_grad_enabled(False)
    testLoader = data.DataLoader(dataset=loader_test, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.n_threads, pin_memory=True)
    index = 0
    for batch, (lr, hr, x) in enumerate(tqdm(testLoader, ncols=80)):
        index = index + 1
        torch.set_grad_enabled(False)
        lr = lr.to(device)
        y = model(lr)
        image3 = y.clamp(0, 1).cpu()
        vutils.save_image(image3,
                          './result/{}/{}{}.png'.format(args.data_name, ''.join(x), index))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    print(torch.cuda.current_device())
    global modele
    net = network.UTVNet().to(device)
    net.load_state_dict(torch.load("./pretrain_model/model_test.pt", map_location=device))
    if args.data_name == 'sRGBSID':
        test_input_dir = './dataset/sRGBSID/test/1/'
        test_input_dir2 = './dataset/sRGBSID/test/2/'
        test_gt_dir = './dataset/sRGBSID/gt/test/'
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '512', args.data_name)

    else:
        test_input_dir = './dataset/ELD/{}/'.format(args.data_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/ELD/{}g/'.format(args.data_name)
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '1024', args.data_name)

    test(net, args, loaderTest, device)


if __name__ == '__main__':
    main()
