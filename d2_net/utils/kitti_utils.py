# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 16:30:00

import numpy as np

class Calibration(object):
    """
    Handles KITTI calibration data and coordinate transformations.
    """
    def __init__(self, calib_filepath):
        self.calib_data = self._read_calib_file(calib_filepath)

    def _read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def get_tr_velo_to_cam(self):
        return self.calib_data['Tr_velo_to_cam'].reshape(3, 4)

    def get_r0_rect(self):
        r0 = np.eye(4)
        r0[:3, :3] = self.calib_data['R0_rect'].reshape(3, 3)
        return r0

    def cart_to_hom(self, pts):
        return np.hstack((pts, np.ones((pts.shape[0], 1))))

    def project_rect_to_velo(self, pts_3d_rect):
        """
        Projects points from rectified camera coordinates to Velodyne coordinates.
        """
        # Inverse of R0_rect
        r0_rect_inv = np.linalg.inv(self.get_r0_rect())
        
        # Inverse of Tr_velo_to_cam
        tr_v2c = self.get_tr_velo_to_cam()
        tr_v2c_inv = np.linalg.inv(np.vstack([tr_v2c, [0, 0, 0, 1]]))
        
        # Project
        pts_3d_ref = self.cart_to_hom(pts_3d_rect) @ r0_rect_inv.T
        pts_3d_velo = pts_3d_ref @ tr_v2c_inv.T
        
        return pts_3d_velo[:, :3]