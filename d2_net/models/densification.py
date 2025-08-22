# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-28 14:30:00

import torch
import torch.nn as nn

class DensificationHead(nn.Module):
    def __init__(self, in_channels, num_points_to_predict=10):
        """
        [THIS IS THE CORRECTED CODE]
        The __init__ function now accepts the 'num_points_to_predict' argument.
        """
        super(DensificationHead, self).__init__()
        
        self.num_predicted_points = num_points_to_predict
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            # The output layer predicts N * 3 values for N points
            nn.Linear(in_channels, 3 * self.num_predicted_points)
        )

    def forward(self, sparse_tensor):
        # sparse_tensor.features has shape (V_total, in_channels)
        offsets = self.predictor(sparse_tensor.features)
        
        # Reshape the output to (V_total, N, 3)
        point_offsets = offsets.view(-1, self.num_predicted_points, 3)
        
        return {
            'sparse_tensor': sparse_tensor,
            'point_offsets': point_offsets 
        }