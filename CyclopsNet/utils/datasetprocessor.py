import torch
import os
import torch.utils.data as data
from PIL import Image
import json as jsonmod
import math
from .utils import get_center, get_object_color, get_text_indices
from concurrent.futures import ProcessPoolExecutor, as_completed
from torchvision import transforms
import itertools
from natsort import natsorted

def process_data(json_path, label_path, image_path):
    """
    Functions that process individual JSON, tag and image files.
    Returns a list of queries, a list of dependencies, a list of objects, a list of states, and a number of objects.
    """
    querys = []
    dependencies = []
    objects = []
    states = []
    count = 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = jsonmod.load(f)
        querys.extend(item["public_description"] for item in data if "public_description" in item)
    
    with open(label_path, "r") as f:
        labels = f.readlines()
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    for label in labels:
        label_parts = label.strip().split(" ")       
        (_, _, _, _, xmin, ymin, xmax, ymax, h, w, l, x, y, z, rotation_y, color, *dependency) = label_parts
 
        dependencies.append(dependency)
        xmin, ymin, xmax, ymax = map(lambda coord: int(float(coord)), [xmin, ymin, xmax, ymax])
        box2d = [xmin, ymin, xmax, ymax]
        
        resized_image = image.crop(box2d).resize((224, 224))
        objects.append(resized_image)

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
        states.append(state)
        count += 1
    
    return querys, dependencies, objects, states, count

class MultiModalDataset(data.Dataset):
    def __init__(self, data_path, dataset, data_split):
        super(MultiModalDataset, self).__init__()
        self.data_path = data_path
        self.dataset = dataset
       
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.objects_list = []
        self.state_list = []
        self.querys_list = []
        self.dependecys_list = []
        self.objects_numbers = []
        self.accumulated_list = []
        self.positive_indexs = []
        self.negative_indexs = []
        
        self.data_dir = os.path.join(self.data_path, self.dataset, data_split)
        
        # Natural sorting with natsorted
        self.json_paths = natsorted([
            os.path.join(self.data_dir, 'jsons', f) 
            for f in os.listdir(os.path.join(self.data_dir, 'jsons')) 
            if f.endswith('.json')
        ])
        self.label_paths = natsorted([
            os.path.join(self.data_dir, 'labels', f) 
            for f in os.listdir(os.path.join(self.data_dir, 'labels')) 
            if f.endswith('.txt')
        ])
        self.image_paths = natsorted([
            os.path.join(self.data_dir, 'images', f) 
            for f in os.listdir(os.path.join(self.data_dir, 'images')) 
            if f.endswith('.jpg')
        ])      
        
        # Parallel processing of all data
        self.process_all_data()

        # Compute the accumulated list
        self.accumulated_list = list(itertools.accumulate(self.objects_numbers))
        flattened_dependencies = list(itertools.chain.from_iterable(self.dependecys_list))
        for i, item in enumerate(flattened_dependencies):
            if item in {'0', '1'}:
                text_indices = get_text_indices(self.accumulated_list, i // 3)
                if text_indices:
                    index_pair = [i // 3, text_indices[i % 3]]
                    if item == '1':
                        self.positive_indexs.append(index_pair)
                    elif item == '0':
                        self.negative_indexs.append(index_pair)
        
        self.length = len(self.positive_indexs)

    def process_all_data(self):
        """
        Process all JSON, tag and image files in parallel using multiple processes.
        """
        with ProcessPoolExecutor() as executor:
            # submit all tasks
            futures = [
                executor.submit(process_data, jp, lp, ip) 
                for jp, lp, ip in zip(self.json_paths, self.label_paths, self.image_paths)
            ]
            for future in as_completed(futures):
                querys, dependencies, objects, states, count = future.result()
                self.querys_list.extend(querys)
                self.dependecys_list.extend(dependencies)
                self.objects_list.extend(objects)
                self.state_list.extend(states)
                self.objects_numbers.append(count)
    
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
    