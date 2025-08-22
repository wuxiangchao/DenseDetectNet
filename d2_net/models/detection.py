# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 12:20:00

import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, tasks_config):
        super(DetectionHead, self).__init__()
        
        shared_channels = 128
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(True),
        )
        
        num_classes = sum(len(task['class_names']) for task in tasks_config)
        num_regression_params = 8
        total_output_channels = num_classes + num_regression_params
        
        self.final_head = nn.Conv2d(shared_channels, total_output_channels, kernel_size=1)

        # 手动初始化最后一层，提供一个稳定的学习起点
        self.final_head.weight.data.normal_(0, 0.01)
        if self.final_head.bias is not None:
            self.final_head.bias.data.zero_()
        # --------------------

    def forward(self, x):
        x = self.shared_conv(x)
        prediction = self.final_head(x)
        return prediction