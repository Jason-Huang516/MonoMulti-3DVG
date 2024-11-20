import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_detection_results(txt_path):
    detections = []
    with open(txt_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 16:
                label = data[0]
                bbox_2d = list(map(float, data[4:8]))  # 2D bbox: [x_min, y_min, x_max, y_max]
                dimensions = list(map(float, data[8:11]))  # 3D dimensions (height, width, length)
                location = list(map(float, data[11:14]))  # 3D location (x, y, z)
                rotation_y = float(data[14])  # Rotation angle
                detections.append({
                    'label': label,
                    'bbox_2d': bbox_2d,
                    'dimensions': dimensions,
                    'location': location,
                    'rotation_y': rotation_y
                })
    return detections


def draw_2d_bbox(ax, bbox_2d, color='r'):
    # bbox_2d 是 [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = bbox_2d
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def main():
    txt_path = 'F:/MonoUNI-main/lib/output/rope3d_eval/data/1632_fa2sd4a11North151_420_1613724070_1613731267_183_obstacle.txt'
    image_path = 'F:/MonoUNI-main/Rope3D_data/image_2/1632_fa2sd4a11North151_420_1613724070_1613731267_183_obstacle.jpg'

    # 读取图像
    image = plt.imread(image_path)

    # 读取检测结果
    detections = load_detection_results(txt_path)

    fig, ax = plt.subplots()
    ax.imshow(image)

    for detection in detections:
        bbox_2d = detection['bbox_2d']

        # 在图像上绘制 2D 检测框
        draw_2d_bbox(ax, bbox_2d)

    plt.show()


if __name__ == "__main__":
    main()
