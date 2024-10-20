import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift
import numpy as np
class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            print('selecting scale fact 8')
            stride = 1
            padding = 1
            kernel_size = 3

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
        
        # print('x ', x.shape)

        # print('self last hidden', self.last_hidden.shape)
        x = torch.cat((x, self.last_hidden), dim=1)

        x = self.compress_in(x)
        # print('compressed x', x.shape)
        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True

class SRFBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(SRFBN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 1
            padding = 2
            kernel_size = 3
        elif upscale_factor == 8:
            stride = 1
            padding = 1
            kernel_size = 3

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor



        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)

        # reconstruction block
		# uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = ConvBlock(num_features, num_features,
                               kernel_size=3, stride=stride, padding=1,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        # self.add_mean = MeanShift(rgb_mean, rgb_std, 1)
        self.interpolate_conv = ConvBlock(1, 1,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        self._reset_state()

        # x = self.sub_mean(x)
		# uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)
		
		# comment for pytorch 0.4.0

        # we are not using bwlow line because our LR and HR sizes are same
        # inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        # print('before interpolate shape', x.shape)
        # inter_res = self.interpolate_conv(x)
        inter_res = x

        # print('Inter res max , LR block', torch.max(inter_res))

        # print('after shape', inter_res.shape)
        # exit()
        # print('Inter_res shape', inter_res.shape)
        x = self.conv_in(x)
        # print('after shape conv_in', x.shape)

        # print('Conv in shape', x.shape)
        x = self.feat_in(x)
        # print('feat_in shape', x.shape)

        ############################## LR block over ##################################

        outs = []
        for step_no in range(self.num_steps):
            
            h = self.block(x)
            # print('h shape', h.shape)
            # print('Out shape', self.out(h).shape)
            temp_out = self.conv_out(self.out(h))
            # print('temp_out', torch.max(temp_out), step_no)
            h = torch.add(inter_res, temp_out)
            # print('inter_res', torch.max(inter_res), step_no)
            # print('h', torch.max(h), step_no)

            # h = self.add_mean(h)
            outs.append(h)
        
        # for idx, i in enumerate(outs):
        #     np.save('{}'.format(idx), i[0][0].detach().cpu().numpy())
        # # exit()
        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()