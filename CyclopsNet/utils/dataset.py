import torch
import os
import torch.utils.data as data
from PIL import Image
import json as jsonmod
import math
from utils import get_center, get_object_color, get_text_indices
from torchvision import transforms
import itertools
from natsort import natsorted

class MultiModalDataset(data.Dataset):
    """
    Load text description, image features and 3D features
    """
    def __init__(self, data_path, dataset, data_split):

        self.data_path = data_path
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.objects_list = []
        self.state_list = []
        self.querys_list = []
        self.dependecys_list = []
        self.objects_numbers = []
        self.accumulated_list = []

        self.positive_indexs = []
        self.negative_indexs = []
        
        self.data_dir = os.path.join(self.data_path, self.dataset, data_split)
        
        self.json_paths = natsorted([os.path.join(self.data_dir, 'jsons', f) for f in os.listdir(os.path.join(self.data_dir, 'jsons')) if f.endswith('.json')])
        self.label_paths = natsorted([os.path.join(self.data_dir, 'labels', f) for f in os.listdir(os.path.join(self.data_dir, 'labels')) if f.endswith('.txt')])
        self.image_paths = natsorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images')) if f.endswith('.jpg')])      

        for json_path, label_path, image_path in zip(self.json_paths, self.label_paths, self.image_paths):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = jsonmod.load(f)
                self.querys_list.extend(item["public_description"] for item in data)
            
            with open(label_path, "r") as f:
                labels = f.readlines()
            image = Image.open(image_path)
            width, height = image.size
            count = 0
            for label in labels:
                label_parts = label.strip().split(" ")
                _, _, _, _, xmin, ymin, xmax, ymax, h, w, l, x, y, z, rotation_y, color = label_parts[:16]
                dependency = label_parts[16:]
                self.dependecys_list.append(dependency)
                xmin, ymin, xmax, ymax = map(lambda coord: int(float(coord)), [xmin, ymin, xmax, ymax])
                box2d = [xmin, ymin, xmax, ymax]

                resized_image = image.crop(box2d).resize((224, 224))   
                self.objects_list.append(resized_image)
                
                h, w, l, x, y, z, rotation_y = map(float, [h, w, l, x, y, z, rotation_y])
                distance = round(math.sqrt(x ** 2 + y ** 2 + z ** 2), 2)
                
                cell_index = get_object_color(box2d, height, width)
                center_x, center_y = get_center(box2d)
                sin_rot_y = math.sin(rotation_y)
                cos_rot_y = math.cos(rotation_y)

                state = [center_x, center_y, 
                         h, w, l,
                        x, y, z, 
                        sin_rot_y, cos_rot_y,
                        distance, 
                        cell_index]
                self.state_list.append(state)
                count += 1
            self.objects_numbers.append(count)
        self.accumulated_list = list(itertools.accumulate(self.objects_numbers))
        flattened_dependencies = list(itertools.chain.from_iterable(self.dependecys_list))
        for i, item in enumerate(flattened_dependencies):
            if item == '1':
                self.positive_indexs.append([i // 3, get_text_indices(self.accumulated_list, i // 3)[i % 3]])
            elif item == '0':
                self.negative_indexs.append([i // 3, get_text_indices(self.accumulated_list, i // 3)[i % 3]])
        self.length = len(self.positive_indexs)
    
    def __getitem__(self, index):
        pos_image = self.transform(self.objects_list[self.positive_indexs[index][0]])
        pos_state = torch.tensor(self.state_list[self.positive_indexs[index][0]])
        pos_query = self.querys_list[self.positive_indexs[index][1]]
        neg_image = self.transform(self.objects_list[self.negative_indexs[index][0]])
        neg_state = torch.tensor(self.state_list[self.negative_indexs[index][0]])
        neg_query = self.querys_list[self.negative_indexs[index][1]]
        return (pos_image, pos_state, pos_query, 1), (neg_image, neg_state, neg_query, 0)

    def __len__(self):
        return self.length

def collate_fn(data):
   
    pos_samples, neg_samples = zip(*data)
    pos_images, pos_states, pos_queries, pos_labels = zip(*pos_samples)
    neg_images, neg_states, neg_queries, neg_labels = zip(*neg_samples)
    images = torch.stack(pos_images + neg_images, dim=0)
    states = torch.stack(pos_states + neg_states, dim=0)
    queries = list(pos_queries) + list(neg_queries) 
    labels = torch.tensor(pos_labels + neg_labels, dtype=torch.float)

    total_samples = images.size(0)
    indices = torch.randperm(total_samples)

    images = images[indices]
    states = states[indices]
    queries = [queries[i] for i in indices]
    labels = labels[indices]

    return images, states, queries, labels

def get_loader(data_path, dataset_path, split_name, batch_size=64, shuffle=True, num_workers=8):
    dataset = MultiModalDataset(data_path, dataset_path, split_name)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=True)
    return data_loader
    