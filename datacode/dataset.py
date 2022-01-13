from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import torchvision.transforms as transform


def get_len(route, phase, format):
    if phase == 'test':
        test_low_data_names = glob(route + '*.{}'.format(format))
        test_low_data_names.sort()
        return len(test_low_data_names), test_low_data_names
    elif phase == 'evaluate':
        metric_low_data_names = glob(route + '*.png')
        metric_low_data_names.sort()
        return len(metric_low_data_names), metric_low_data_names
    else:
        return 0, []


class rgbDataset(Dataset):
    def __init__(self, route, route2, gtroute, phase, patch_size, data_name):
        self.route = route
        self.gtroute = gtroute
        self.route2 = route2
        self.phase = phase
        self.dataname = data_name
        self.patch_size = patch_size
        self.input_images = [None] * 500
        self.gt_images = [None] * 250
        self.num = [0] * 250
        self.pre = [0] * 250
        if self.dataname == 'sRGBSID':
            self.len, self.low_names = get_len(route, phase, 'png')
            self.len2, self.low_names2 = get_len(route2, phase, 'png')
            self.length = self.len + self.len2
            self.names = self.low_names + self.low_names2
            print(self.length)
        else:
            self.len, self.low_names = get_len(self.route, phase, 'JPG')
            self.length = self.len
            self.names = self.low_names

    def __getitem__(self, index):
        name = os.path.basename(self.names[index])[0:-4]
        if self.dataname == 'sRGBSID':
            id = os.path.basename(self.names[index])[0:5]
            gtdir = glob(self.gtroute + id + '_00_*.png')[0]
        elif self.dataname == 'ELD_cano':
            id = os.path.basename(self.names[index])[0:6]
            gtdir = glob(self.gtroute + id + '*.JPG')[0]
        elif self.dataname == 'ELD_sony':
            id = os.path.basename(self.names[index])[0:6]
            gtdir = glob(self.gtroute + id + '*.JPG')[0]
        elif self.dataname == 'ELD_niko':
            id = os.path.basename(self.names[index])[0:12]
            gtdir = glob(self.gtroute + id + '*.JPG')[0]
        else:
            return 0, ''

        low_im = Image.open(self.names[index])
        high_im = Image.open(gtdir)
        if self.patch_size == '1024':
            low_im = low_im.resize((1024, 768))
            high_im = high_im.resize((1024, 768))
        else:
            low_im = low_im.resize((512, 384))
            high_im = high_im.resize((512, 384))
        trainImage, groundTruth = low_im, high_im
        return transform.ToTensor()(trainImage), transform.ToTensor()(groundTruth), name

    def __len__(self):
        return self.length
