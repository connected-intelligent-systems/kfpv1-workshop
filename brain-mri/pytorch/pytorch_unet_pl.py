from kfp import dsl
from kfp.dsl import Input, Output, HTML, Artifact, Model, Dataset
from kfp import kubernetes

PL_NAME = 'pytorch_unet'

    
@dsl.component(base_image='harbor.foresight-next.plaiful.org/kfp-examples/ultralytics', packages_to_install=['mlflow'])
def train_unet(TRAIN_IMG_DIR: str, TRAIN_MASK_DIR: str, VAL_IMG_DIR: str, VAL_MASK_DIR: str,
               IMG_HEIGHT: int = 256, IMG_WIDTH: int = 256, BATCH_SIZE: int = 8, NUM_WORKERS: int = 2, LEARNING_RATE: float = 1e-4,
               NUM_EPOCHS: int = 3, LOAD_MODEL: bool = False, PIN_MEMORY: bool = True, mount_path: str = ''):
    import torch
    import torch.nn as nn
    import torchvision.transforms.functional as TF
    import torchvision
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from tqdm import tqdm
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from PIL import Image
    import os
    import numpy as np
    from torch.utils.data import Dataset
    import mlflow
    
    class Dataset(Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform = transform
            self.images = os.listdir(image_dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask 
    
    
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class UNET(nn.Module):
        def __init__(
                self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
        ):
            super(UNET, self).__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Down part of UNET
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature

            # Up part of UNET
            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))

            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skip_connections = []

            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

            return self.final_conv(x)
   
    
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def load_checkpoint(checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])

    def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
    ):
        train_ds = Dataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        val_ds = Dataset(
            image_dir=val_dir,
            mask_dir=val_maskdir,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        return train_loader, val_loader

    def check_accuracy(loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        model.train()

    def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
    ):
        model.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/truth_{idx}.png")
            torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")

        model.train()
    
    
    def train_fn(loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
    
    # MlFlow setup
    mlflow.tracking.set_tracking_uri('http://mlf-mlflow.kubeflow.svc.cluster.local:5000')
    experiment = mlflow.set_experiment("pytorch-unet-brain-mri")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_dir = os.path.join(mount_path, 'saved_images')
        os.makedirs(save_dir, exist_ok=True) 
        save_predictions_as_imgs(
            val_loader, model, folder=save_dir, device=DEVICE
        ) 
    

@dsl.component(base_image='harbor.foresight-next.plaiful.org/kfp-examples/ultralytics')
def visualize_validation_batches(mount_path: str,
                                 truth_batch_name: str,
                                 pred_batch_name: str,
                                 original_batch_name: str,
                                 image_visualization: Output[HTML]):
    import json
    import cv2
    import os
    import base64
  

    def encode_img(img, im_type):
        success, encoded_img = cv2.imencode('.{}'.format(im_type), img)
        if success:
            return base64.b64encode(encoded_img).decode()
        return ''
  
    def img_to_base64(image_path):
        img = cv2. imread(image_path)   
        encoded_img = encode_img(img, 'jpg')
        b64_src = 'data:image/jpeg;base64,'
        img_src = b64_src + encoded_img
        return img_src
  
    image_path = os.path.join(mount_path, 'saved_images')
  
    original_batch = img_to_base64(os.path.join(image_path, original_batch_name))
    val_batch = img_to_base64(os.path.join(image_path, truth_batch_name))
    pred_batch = img_to_base64(os.path.join(image_path, pred_batch_name))
   
    html = '<!DOCTYPE html><html><body><h2>Ground truth</h2><img src='+val_batch+' alt="ground truth"></body><body><h2>Prediction</h2><img src='+pred_batch+' alt="prediction"></body><body><h2>Original</h2><img src='+original_batch+' alt="original"></body></html>' 
    
    with open(image_visualization.path, 'w') as metadata_file:
        json.dump(html, metadata_file)   

   
@dsl.pipeline(name=PL_NAME)   
def image_segmentation(
    pvc_name: str = 'workshoptest-volume', 
    TRAIN_IMG_DIR: str = '/usr/share/volume/yolo/datasets/yolo_mri_brain/train/images',
    TRAIN_MASK_DIR: str = '/usr/share/volume/yolo/datasets/yolo_mri_brain/train/masks',
    VAL_IMG_DIR: str = '/usr/share/volume/yolo/datasets/yolo_mri_brain/valid/images',
    VAL_MASK_DIR: str = '/usr/share/volume/yolo/datasets/yolo_mri_brain/valid/masks',
    IMG_HEIGHT: int = 256, 
    IMG_WIDTH: int = 256, 
    BATCH_SIZE: int = 8, 
    NUM_WORKERS: int = 2,
    LEARNING_RATE: float = 1e-4,
    NUM_EPOCHS: int = 3,
    PIN_MEMORY: bool = True,
    LOAD_CHECKPOINT: bool = False,
    mount_path: str = '/usr/share/volume'
):  

     
    learner = train_unet(TRAIN_IMG_DIR=TRAIN_IMG_DIR, TRAIN_MASK_DIR=TRAIN_MASK_DIR, VAL_IMG_DIR=VAL_IMG_DIR, VAL_MASK_DIR=VAL_MASK_DIR, IMG_HEIGHT=IMG_HEIGHT, 
                         IMG_WIDTH=IMG_WIDTH, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, LEARNING_RATE=LEARNING_RATE, NUM_EPOCHS=NUM_EPOCHS, 
                         LOAD_MODEL=LOAD_CHECKPOINT, PIN_MEMORY=PIN_MEMORY, mount_path=mount_path).set_accelerator_limit(1).set_accelerator_type("nvidia.com/gpu")


    visualize_val = visualize_validation_batches(mount_path=mount_path, truth_batch_name='pred_0.png', pred_batch_name='truth_0.png', original_batch_name='image_0.png').after(learner) # type: ignore

    kubernetes.mount_pvc(
        learner,
        pvc_name=pvc_name,
        mount_path='/usr/share/volume',
    )
    kubernetes.mount_pvc(
        visualize_val,
        pvc_name=pvc_name,
        mount_path='/usr/share/volume',
    )

if __name__ == '__main__':
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace="workshop", pl_name='pytorch_unet')
    helper.upload_pipeline(pipeline_function=image_segmentation)
    #helper.create_run(pipeline_function=image_segmentation) 