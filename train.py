from dataset import coco_dataset
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
writer = SummaryWriter()

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)
#==============================================================================#
#=============================== USER CONFIGURATION ===========================#
#==============================================================================#

# path to data
train_data_dir = "./images/train2017"
train_anno = "./annotations/instances_train2017.json"

val_data_dir = "./images/val2017"
val__anno = "./annotations/instances_val2017.json"

# Batch size
train_batch_size = 2

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
num_classes = 2
num_epochs = 10

lr = 0.005
momentum = 0.9
weight_decay = 0.005


#==============================================================================#
#=============================== USER CONFIGURATION ===========================#
#==============================================================================#

if __name__ == "__main__":

    # create own Dataset
    train_ds = coco_dataset(
        train_data_dir, train_anno, transforms=get_transform()
    )
    # print(len(train_ds.ids))
    # print(len(train_ds.filtered_ids))
    # exit()

    # own DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=train_shuffle_dl,
        num_workers=num_workers_dl,
        collate_fn=collate_fn,
    )



    # # DataLoader is iterable over Dataset
    # for imgs, annotations in train_loader:
    #     imgs = list(img.to(device) for img in imgs)
    #     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #     print(annotations)


    model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    len_trainloader = len(train_loader)


    # Training
    for epoch in range(num_epochs):
        global_step = epoch*len_trainloader
        print(f"Epoch: {epoch}/{num_epochs}")
        model.train()
        
        for idx, (imgs, annotations) in enumerate(train_loader):
            
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            writer.add_scalar('Loss/train',losses.item(), global_step + idx)
            print(f"Iteration: {idx}/{len_trainloader}, Loss: {losses}")