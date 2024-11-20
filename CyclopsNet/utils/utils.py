import re
import bisect
import numpy as np
from scipy.spatial import ConvexHull

def lr_lambda(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    else:
        return 1.0
    
def get_object_color(box2d, height, width):
    xmin, ymin, xmax, ymax = map(int, box2d)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    grid_width = width // 3
    grid_height = height // 3
    x_index = int(center_x // grid_width)
    y_index = int(center_y // grid_height)
    cell_index = y_index * 3 + x_index
    return int(cell_index)

def get_center(box2d):
    xmin, ymin, xmax, ymax = map(int, box2d)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    return center_x, center_y

def split_text_by_word(text):
    """
    将文本按单词和数字分块，保留整个单词（包括内部的引号等符号），
    例如将 "I’m 1.5m." 分块为 ["I’m", "1.5m"]。

    Args:
        text (str): 要分块的文本。

    Returns:
        list: 文本块列表。
    """
    # 使用正则表达式匹配单词或数字，保留引号等符号
    words = re.findall(r"[a-zA-Z0-9'’]+(?:\.\d+)?", text)
    return words

def img_place(cell_index):
    mapping = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right"]
    return mapping[cell_index]


def get_text_indices(cumulative_list, target_index):
    """
    使用二分搜索查找 target_index 所在的区间，并返回对应的三个文本描述索引。

    :param cumulative_list: 累加列表，假设为递增的整数列表。
    :param target_index: 目标索引。
    :return: 包含三个文本描述索引的列表，或 None（如果 target_index 超出范围）。
    """
    # 使用 bisect_right 找到第一个大于 target_index 的位置
    i = bisect.bisect_right(cumulative_list, target_index)
    
    if i < len(cumulative_list):
        text_start_index = i * 3
        return [text_start_index, text_start_index + 1, text_start_index + 2]
    else:
        return None

def get_3d_box_corners(box):
    x, y, z, h, w, l, rotation_y = box
    # 计算旋转矩阵
    cos_r = np.cos(rotation_y)
    sin_r = np.sin(rotation_y)
    R = np.array([[cos_r, 0, sin_r],
                  [0, 1, 0],
                  [-sin_r, 0, cos_r]])

    # 计算初始8个顶点（未旋转）
    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = h / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = w / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # 合并为顶点矩阵
    corners = np.vstack((x_corners, y_corners, z_corners))

    # 旋转并平移到目标位置
    corners_3d = np.dot(R, corners)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    return corners_3d.T  # 返回形状为 (8, 3) 的顶点坐标

def compute_3d_iou(pred_box, gt_box):
    pred_corners = get_3d_box_corners(pred_box)
    gt_corners = get_3d_box_corners(gt_box)

    # 使用 ConvexHull 求交集和并集
    try:
        pred_hull = ConvexHull(pred_corners)
        gt_hull = ConvexHull(gt_corners)
        all_corners = np.vstack((pred_corners, gt_corners))
        union_hull = ConvexHull(all_corners)

        pred_volume = pred_hull.volume
        gt_volume = gt_hull.volume
        intersection_volume = max(0.0, pred_volume + gt_volume - union_hull.volume)
        union_volume = union_hull.volume

        iou = intersection_volume / union_volume
    except:
        # 如果ConvexHull计算失败（例如点共线或共面），则设置IoU为0
        iou = 0.0

    return iou

def compute_f1_score(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    matched_pred = set()

    # 计算TP和FP
    for pred_idx, pred_box in enumerate(pred_boxes):
        matched = False
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_3d_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                matched = True
                break
        if not matched:
            fp += 1

    # 计算FN
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in matched_gt:
            fn += 1

    # 计算Precision, Recall, F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1_score
