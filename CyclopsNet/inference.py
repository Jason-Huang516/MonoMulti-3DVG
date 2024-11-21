import math
import torch
import os
from model import CyclopsNet

from PIL import Image
import json as jsonmod
from torchvision import transforms
import arguments
from utils.utils import get_object_color, get_center, compute_f1_score
from natsort import natsorted
from tqdm import tqdm


def process_nested_list(nested_list):
    processed_lists = []
    for sublist in nested_list:
        processed_sublist = []

        for item in sublist:
            formatted_item = item.replace("[", "").replace("]", "").replace("'", "").replace(",", "").strip()
            processed_sublist.append(formatted_item)

        processed_lists.append(processed_sublist)
    return processed_lists

def load_model(weights_path, parser, device):
    model = CyclopsNet(parser).to(device)
    checkpoint = torch.load(weights_path, map_location=parser.device)
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, epoch_loss {})"
          .format(parser.resume, epoch, epoch_loss))
    model.eval()
    return model

def parse_label_string(label_str, use_IoU):
    """Parse label string and extract box2d"""
    label_str = label_str.strip()
    if label_str.startswith("[") and label_str.endswith("]"):
        # Parsing gt_label
        label_parts = label_str.split(",")
        label_parts = [part.strip().strip("'") for part in label_parts]
        if use_IoU:
            h, w, l, x, y, z, ry  = [float(coord) for coord in label_parts[9:16]]
            return {'box3d': [float(x), float(y), float(z), float(h), float(w), float(l), float(ry)]}
        else:
            xmin, ymin, xmax, ymax = [float(coord) for coord in label_parts[4:8]]
            return {'box2d': [xmin, ymin, xmax, ymax]}
    else:
        # Parsing result_target label
        label_parts = label_str.strip().split(" ")
        if use_IoU:
            h, w, l, x, y, z, ry  = [float(coord) for coord in label_parts[8:15]]
            return {'box3d': [float(x), float(y), float(z), float(h), float(w), float(l), float(ry)]}
        else:
            xmin, ymin, xmax, ymax = [float(coord) for coord in label_parts[4:8]]
            return {'box2d': [float(xmin), float(ymin), float(xmax), float(ymax)]}

def boxes_are_close(box1, box2, tol=1.0):
    return all(abs(a - b) <= tol for a, b in zip(box1, box2))

def test(model, data_list, threshold, device, save_dir, use_IoU):
    os.makedirs(save_dir, exist_ok=True)

    total_FP = 0  
    total_FN = 0  
    total_TP = 0  
    
    for image_data in tqdm(data_list, desc="Processing"):
        descriptions = image_data['public_descriptions']
        targets = image_data['targets']
        filtered_targets = [target for target in targets if len(target['image_feature']) > 0]
        if len(filtered_targets) == 0:
            continue
        image_features = torch.stack([target['image_feature'] for target in filtered_targets], dim=0).to(device)
        states = torch.tensor([target['state'] for target in targets], device=device)
        gt_labels = image_data['gt']  
        labels = image_data['labels']  

        FP = 0  
        FN = 0  
        TP = 0  
        
        with torch.no_grad():
            for i in range(3):
                logits = model(image_features.to('cuda'), states.to('cuda'), [descriptions[i]] * len(image_features))[-2]                    
                probabilities = torch.sigmoid(logits)
                result_targets = []

                for idx, prob in enumerate(probabilities):
                    similarity = prob.item()
                    if similarity > threshold:
                        result_targets.append(labels[idx])

                result_boxes = []
                for result_target in result_targets:
                    result_info = parse_label_string(result_target, use_IoU)
                    if result_info is not None:
                        if use_IoU:
                            result_boxes.append(result_info['box3d'])
                        else:
                            result_boxes.append(result_info['box2d'])

                gt_boxes = []
                for gt_label in gt_labels[i]:
                    gt_info = parse_label_string(gt_label, use_IoU)
                    if gt_info is not None:
                        if use_IoU:
                            gt_boxes.append(gt_info['box3d'])
                        else:
                            gt_boxes.append(gt_info['box2d'])
                        
                if use_IoU:
                    tp, fp, fn, _, _, _ = compute_f1_score(result_boxes, gt_boxes, iou_threshold=0.7)
                    TP += tp
                    FP += fp
                    FN += fn
                else:
                    for result_box in result_boxes:
                        if result_box in gt_boxes:
                            TP += 1  
                        else:
                            FP += 1  

                    for gt in gt_boxes:
                        if gt not in result_boxes:
                            FN += 1

        total_TP += TP
        total_FP += FP
        total_FN += FN
        
    return total_FP, total_FN, total_TP

def main():
    parser = arguments.get_argument_parser().parse_args()  
    use_IoU = True
    weights_path = "training/model/model_50epoch_checkpoint.pth.tar"
    image_dir = "data/rope3d/test/images"
    if use_IoU:
        label_dir = "data/rope3d/test/bbox3d"
    else:
        label_dir = "data/rope3d/test/labels"
    json_dir = "data/rope3d/test/jsons"
    save_dir = "results"
    threshold = 0.6  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(weights_path, parser, device)

    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    data_list = []

    image_paths = natsorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    json_paths = natsorted([os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')])
    label_paths = natsorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')])

    for json_path, image_path, label_path in tqdm(zip(json_paths, image_paths, label_paths)):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = jsonmod.load(f)
        gt = process_nested_list([item["label_3"] for item in data])
        public_descriptions = [item["public_description"] for item in data]
        with open(label_path, "r") as f:
            labels = f.readlines()

        with Image.open(image_path) as image:
            width, height = image.size

            targets = []
            labels_list = []
            for label in labels:
                label_parts = label.strip().split(" ")
                _, _, _, _, xmin, ymin, xmax, ymax, h, w, l, x, y, z, rotation_y = label_parts[:15]
                
                box2d = [int(float(coord)) for coord in [xmin, ymin, xmax, ymax]]
                cropped_image = image.crop(box2d).resize((224, 224))
                image_feature = transform(cropped_image)
                
                h, w, l, x, y, z, rotation_y = map(float, [h, w, l, x, y, z, rotation_y])
                distance = round(math.sqrt(x ** 2 + y ** 2 + z ** 2), 2)
                cell_index = get_object_color(box2d, height, width)
                
                center_x, center_y = get_center(box2d)
                sin_rot_y = math.sin(rotation_y)
                cos_rot_y = math.cos(rotation_y)
                
                state = [
                    center_x, center_y,
                    h, w, l,
                    x, y, z,
                    sin_rot_y, cos_rot_y,
                    distance,
                    cell_index
                ]
                
                targets.append({
                    'image_feature': image_feature,
                    'state': state
                })
                labels_list.append(label)
        
        image_data = {
            'image_path': image_path,
            'public_descriptions': public_descriptions,
            'targets': targets,
            'labels': labels_list,
            'gt': gt
        }
        data_list.append(image_data)

    total_FP, total_FN, total_TP = test(model, data_list, threshold, device, save_dir, use_IoU)
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"FP: {total_FP}")
    print(f"FN: {total_FN}")
    print(f"TP: {total_TP}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")

if __name__ == '__main__':
    main()
