import os

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import sys
sys.path.append("./cocoapi/PythonAPI")

from pycocotools.coco import COCO

class coco_dataset(Dataset):
    def __init__(self, root_path, annos_path, transforms = None) -> None:
        """
        :param:
        """
        super().__init__()
        self.root = root_path
        self.coco = COCO(annos_path)
        self.transforms = transforms


        self.ids = list(sorted(self.coco.imgs.keys()))
        # filter images with emtpy boxes
        self.ids = [key for key in self.ids if self._filter_empty_boxes(key)]

    def _filter_empty_boxes(self, img_id):

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        if len(coco_annotation) == 0:
            return False
        else:
            return True


    def __getitem__(self, index: int):
        
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)
        # path for input image
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # labels = torch.tensor([anno["category_id"] for anno in coco_annotation])
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self) -> int:
        return len(self.ids)




if __name__ == "__main__":
    coco = COCO("./annotations/instances_val2017.json")
    print(len(coco.imgs.keys()))