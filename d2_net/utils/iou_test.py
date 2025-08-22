import torch
from iou_wrapper import rotated_box_iou

def test_iou():
    a = torch.tensor([[0, 0, 2, 2, 0.0]], dtype=torch.float32).cuda()
    b = torch.tensor([[0, 0, 2, 2, 0.0]], dtype=torch.float32).cuda()
    print("IoU (same box):", rotated_box_iou(a, b).cpu().numpy())  #  ≈ 1

    c = torch.tensor([[5, 5, 2, 2, 0.0]], dtype=torch.float32).cuda()
    print("IoU (separate boxes):", rotated_box_iou(a, c).cpu().numpy())  #  ≈ 0

    d = torch.tensor([[0, 0, 2, 2, 1]], dtype=torch.float32).cuda()
    print("IoU (rotated 90°):", rotated_box_iou(a, d).cpu().numpy())  #  0 ~ 1 

if __name__ == "__main__":
    test_iou()
