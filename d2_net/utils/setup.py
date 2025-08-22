# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-08-20 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-21 17:30:11

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rotated_iou_cuda',
    ext_modules=[
        CUDAExtension(
            'rotated_iou_cuda_kernel', # 编译后的模块名
            ['iou_cuda.cu'] # 源文件名
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)