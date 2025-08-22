/*
* -*- coding: utf-8 -*-
* @Author: WUXIANGCHAO
* @Date: 2025-08-17 22:15:30
* @Last Modified by: WUXIANGCHAO
* @Last Modified time: 2025-08-20 15:45:00
*/


#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>

// 定义一个简单的点结构
struct Point {
    float x, y;
};

// 计算两个点的叉积
__device__ inline float cross_product(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

// 计算点p相对于线段ab的位置
__device__ inline float get_orientation(Point a, Point b, Point p) {
    return cross_product({b.x - a.x, b.y - a.y}, {p.x - a.x, p.y - a.y});
}

// 计算两条线段(a-b, c-d)的交点
__device__ Point get_intersection(Point p1, Point p2, Point p3, Point p4) {
    float x1 = p1.x, y1 = p1.y;
    float x2 = p2.x, y2 = p2.y;
    float x3 = p3.x, y3 = p3.y;
    float x4 = p4.x, y4 = p4.y;

    float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if (fabsf(den) < 1e-7) {
        return {NAN, NAN}; // Lines are parallel or collinear
    }

    float t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    float u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3));

    float t = t_num / den;
    float u = u_num / den;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        return {x1 + t * (x2 - x1), y1 + t * (y2 - y1)};
    }

    return {NAN, NAN}; // Intersection is outside the segments
}

// Sutherland-Hodgman 算法核心：用一条裁剪边(clip_p1 -> clip_p2)来裁剪一个多边形
__device__ void clip_polygon(Point* subject_polygon, int& subject_n, Point clip_p1, Point clip_p2, Point* temp_polygon) {
    int temp_n = 0;
    for (int i = 0; i < subject_n; ++i) {
        Point p1 = subject_polygon[i];
        Point p2 = subject_polygon[(i + 1) % subject_n];

        float orient_p1 = get_orientation(clip_p1, clip_p2, p1);
        float orient_p2 = get_orientation(clip_p1, clip_p2, p2);

        if (orient_p1 >= 0) { // p1 在裁剪区域内
            temp_polygon[temp_n++] = p1;
        }

        // p1和p2在裁剪边两侧，计算交点
        if ((orient_p1 * orient_p2) < 0) {
            Point intersection = get_intersection(p1, p2, clip_p1, clip_p2);
            // Only add valid, non-NaN intersection points ---
            if (!isnan(intersection.x)) {
                temp_polygon[temp_n++] = intersection;
            }
        }
    }
    // 更新被裁剪的多边形
    subject_n = temp_n;
    for (int i = 0; i < temp_n; ++i) {
        subject_polygon[i] = temp_polygon[i];
    }
}

// 计算多边形的面积 (Shoelace formula)
__device__ float polygon_area(Point* polygon, int n) {
    if (n < 3) return 0.0f;
    float area = 0.0f;
    for (int i = 0; i < n; ++i) {
        area += cross_product(polygon[i], polygon[(i + 1) % n]);
    }
    return abs(area) / 2.0f;
}

__global__ void rotated_iou_kernel(
    const float* __restrict__ corners1, // (N, 4, 2)
    const float* __restrict__ corners2, // (M, 4, 2)
    float* __restrict__ intersection_areas, // (N, M)
    int N, int M) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * M) return;

    int n = index / M;
    int m = index % M;

    // 提取两个box的角点
    Point box1_corners[4];
    Point box2_corners[4];
    for(int i=0; i<4; ++i){
        box1_corners[i] = {corners1[(n*4+i)*2], corners1[(n*4+i)*2+1]};
        box2_corners[i] = {corners2[(m*4+i)*2], corners2[(m*4+i)*2+1]};
    }
    
    // 使用Sutherland-Hodgman算法
    // intersection_polygon可以最多有8个顶点
    Point intersection_polygon[8];
    Point temp_polygon[8];
    int intersection_n = 4;
    for(int i=0; i<4; ++i) intersection_polygon[i] = box1_corners[i];
    
    for(int i=0; i<4; ++i){
        if(intersection_n == 0) break;
        clip_polygon(intersection_polygon, intersection_n, box2_corners[i], box2_corners[(i+1)%4], temp_polygon);
    }
    
    intersection_areas[index] = polygon_area(intersection_polygon, intersection_n);
}

// 主C++函数，作为Python的接口
at::Tensor calculate_intersection_areas(at::Tensor corners1, at::Tensor corners2) {
    TORCH_CHECK(corners1.is_cuda(), "corners1 must be a CUDA tensor");
    TORCH_CHECK(corners2.is_cuda(), "corners2 must be a CUDA tensor");
    TORCH_CHECK(corners1.dim() == 3 && corners1.size(1) == 4 && corners1.size(2) == 2, "corners1 shape must be (N, 4, 2)");
    TORCH_CHECK(corners2.dim() == 3 && corners2.size(1) == 4 && corners2.size(2) == 2, "corners2 shape must be (M, 4, 2)");

    int N = corners1.size(0);
    int M = corners2.size(0);

    auto options = corners1.options();
    auto intersection_areas = at::zeros({N, M}, options);
    
    int total_pairs = N * M;
    if(total_pairs == 0) return intersection_areas;
    
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    rotated_iou_kernel<<<blocks, threads>>>(
        corners1.data_ptr<float>(),
        corners2.data_ptr<float>(),
        intersection_areas.data_ptr<float>(),
        N, M);
        
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return intersection_areas;
}

// 绑定到Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_intersection", &calculate_intersection_areas, "Calculate intersection areas of rotated boxes (CUDA)");
}