import torch.utils.data as data
import os
import numpy as np
from data import common
import torch
import torchvision.transforms as transforms

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        print('Inside data loader')
        print(self.opt)
        self.train = (opt['phase'] == 'train')
        path = opt['data_path']
        print('path', path)
        if(self.train):
            self.file_names = np.load(os.path.join(path, 'train_files.npy'))
        else:
            self.file_names = np.load(os.path.join(path, 'val_files.npy'))

        self.hr_dem = os.path.join(path, 'HR_DEM')
        self.lr_dem = os.path.join(path, 'LR_DEM')
        
        self.makeTensor = transforms.ToTensor()

    def __getitem__(self, index):
        name = self.file_names[index] + '.dem'
        # print('input name', name)
        
        LR_DEM = np.loadtxt(os.path.join(self.lr_dem, name))
        HR_DEM = np.loadtxt(os.path.join(self.hr_dem, name))
        # print('LR_DEM shape', LR_DEM.shape)
        #print('HRshape', HR_DEM.shape)
        LR_DEM = LR_DEM[..., np.newaxis]
        HR_DEM = HR_DEM[..., np.newaxis]
        # print('Input shapes, ', LR_DEM.shape)
        # print('HR_DEM', HR_DEM.shape)
        LR_DEM, HR_DEM = self._get_patch(LR_DEM, HR_DEM)
        LR_DEM, HR_DEM = common.np2Tensor([LR_DEM, HR_DEM], self.opt['rgb_range'])

        # LR_DEM = LR_DEM.transpose(2,0,1)
        # HR_DEM = HR_DEM.transpose(2,0,1)

        # print('LR_DEM shape', LR_DEM.shape)
        # LR_DEM = LR_DEM.repeat(3,2).transpose([2,0,1])
        # HR_DEM = HR_DEM.repeat(3,2).transpose([2,0,1])
        # LR_DEM = Image.fromarray(LR_DEM)
        # HR_DEM = Image.fromarray(HR_DEM)

        # apply the same transform to both A and B
        # transform = transforms.ToTensor()

        # LR_DEM = torch.tensor(LR_DEM)
        # HR_DEM = torch.tensor(HR_DEM)

        # LR_DEM = self.makeTensor(LR_DEM)
        # HR_DEM = self.makeTensor(HR_DEM)
        # fractal = transform(self.noise)
        # if using instance maps

        input_dict = {'LR': LR_DEM,
                      'HR': HR_DEM,
                      'LR_path': name,
                      'HR_path' : name
                     }

        # Give subclasses a chance to modify the final output
        #print('Able to return data')
        return input_dict

    def __len__(self):
        return len(self.file_names)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(
            lr, hr, LR_size, self.scale)
        # print('shapes again here', lr.shape, hr.shape)
        lr, hr = common.augment([lr, hr])
        # lr = common.add_noise(lr, self.opt['noise'])
        # print('shapes again', lr.shape, hr.shape)
        return lr, hr
