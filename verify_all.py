# verify_all.py

import torch

print("--- 1. PyTorch & CUDA Verification ---")
try:
    print(f"PyTorch Version: {torch.__version__}")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {is_cuda_available}")
    assert is_cuda_available, "CUDA not available to PyTorch"
    print(f"CUDA Version Detected by PyTorch: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch verification PASSED.\n")
except Exception as e:
    print(f"❌ PyTorch verification FAILED: {e}\n")


print("--- 2. spconv Functional Test ---")
try:
    import spconv.pytorch as spconv
    features = torch.randn(5, 3).cuda()
    indices = torch.randint(0, 10, (5, 4), device='cuda').int()
    indices[:, 0] = 0
    sp_tensor = spconv.SparseConvTensor(features, indices, [10, 10, 10], 1)
    print("✅ spconv verification PASSED.\n")
except Exception as e:
    print(f"❌ spconv verification FAILED: {e}\n")


print("--- 3. Pytorch3D Functional Test ---")
try:
    # 尝试导入pytorch3d的一个核心CUDA功能
    from pytorch3d.ops import knn_points
    points1 = torch.randn(1, 10, 3).cuda()
    points2 = torch.randn(1, 20, 3).cuda()
    # 执行一个简单的KNN操作，这将调用其CUDA内核
    knn_res = knn_points(points1, points2, K=1)
    print("✅ Pytorch3D verification PASSED.\n")
except ImportError as e:
    print(f"❌ Pytorch3D verification FAILED with ImportError: {e}")
    print("This is likely a library path or version mismatch issue.")
except Exception as e:
    print(f"❌ Pytorch3D verification FAILED with other error: {e}")