from kfp import dsl
from kfp.components import create_component_from_func, OutputPath, InputPath
from typing import NamedTuple
from kfp.onprem import mount_pvc

PL_NAME = 'fastai_resnet34'


def show_prediction(learner: InputPath(), DIR_IMGS: str = '', DIR_MASKS: str = '', CODES: list = [], IMG_SIZE: int = 256, BATCH_SIZE: int = 32, NUM_WORKERS: int = 0, LR: float = 0.01):
    from fastai.vision.all import unet_learner, BCEWithLogitsLossFlat, get_image_files, RandomSplitter, Normalize, RatioResize, ImageBlock, MaskBlock, Path, DataBlock, imagenet_stats
    import torchvision.models as models
    import matplotlib as mpl
    import numpy as np
    
    def accuracy(pred, target):
        target = target.squeeze(1)
        return (pred.argmax(dim=1) == target).float().mean()

    def IoU(pred, targs, maxClassValue=1):
        n = targs.shape[0]
        pred = pred.argmax(dim=1).view(n,-1)
        targs = targs.view(n, -1) // maxClassValue
        intersect = (pred * targs).sum().float()
        union = (pred + targs).sum().float()
        return intersect / (union - intersect + 1.0)

    def dice(pred, targs):
        pred = (pred>0).float()
        return 2. * (pred*targs).sum() / (pred+targs).sum()   
    def colormap():
        cpts = [0.0, 254.0/255.0, 1.0]
        colors = [(cpts[0], (0, 0, 0)), (cpts[1], (.5, .5, .5)), (cpts[2], (1, 0, 0))]
        return mpl.colors.LinearSegmentedColormap.from_list('cmap', colors)
    
    
    paths_imgs = get_image_files(DIR_IMGS)
    paths_masks = get_image_files(DIR_MASKS)

    assert len(paths_imgs)>0
    assert len(paths_masks)>0
    assert len(paths_imgs)==len(paths_masks)

    codes = np.array(CODES)

    def label_function(filename):
           return DIR_MASKS / Path(filename.stem + filename.suffix)


    datablocks = DataBlock(blocks = (ImageBlock, MaskBlock(codes)),
                                     get_items = get_image_files,
                                     get_y = label_function,
                                     splitter = RandomSplitter(),
                                     item_tfms = RatioResize(IMG_SIZE),
                                     batch_tfms = [Normalize.from_stats(*imagenet_stats)])
    
    dl = datablocks.dataloaders(DIR_IMGS, bs=BATCH_SIZE, num_workers=NUM_WORKERS)
    dl.vocab = codes
    
    MODEL_BACKBONE = models.resnet34
    learn = unet_learner(dl, MODEL_BACKBONE, n_in=3, n_out=1, lr=LR, loss_func=BCEWithLogitsLossFlat(), metrics=[accuracy, dice, IoU])
    with open(learner, 'rb') as model_file:
        learn.load(model_file)
    plot = learn.show_results(max_n=4, figsize=[12,6], vmin=0, vmax=1, cmap=colormap())
    print(plot)
    print(type(plot))
    print(dir(plot))
    
     
    
def train_unet(learner: OutputPath(), DIR_IMGS: str = '', DIR_MASKS: str = '', CODES: list = [], IMG_SIZE: int = 256, BATCH_SIZE: int = 16, NUM_WORKERS: int = 0, LR: float = 0.01, EPOCHS: int = 10):
    
    from fastai.vision.all import get_image_files, DataBlock, ImageBlock, MaskBlock, RandomSplitter, RatioResize, Normalize, imagenet_stats, Path
    from fastai.vision.all import unet_learner, BCEWithLogitsLossFlat, accuracy_multi, ShowGraphCallback, CSVLogger
    import torchvision.models as models
    import numpy as np
    import torch
    import mlflow
   
    # MlFlow setup
    mlflow.tracking.set_tracking_uri('http://mlflow-server:5000')
    experiment = mlflow.set_experiment("fastai-brain-mri") 
    
    def accuracy(pred, target):
        target = target.squeeze(1)
        return (pred.argmax(dim=1) == target).float().mean()

    def IoU(pred, targs, maxClassValue=1):
        n = targs.shape[0]
        pred = pred.argmax(dim=1).view(n,-1)
        targs = targs.view(n, -1) // maxClassValue
        intersect = (pred * targs).sum().float()
        union = (pred + targs).sum().float()
        return intersect / (union - intersect + 1.0)

    def dice(pred, targs):
        pred = (pred>0).float()
        return 2. * (pred*targs).sum() / (pred+targs).sum()   
    
    
    paths_imgs = get_image_files(DIR_IMGS)
    paths_masks = get_image_files(DIR_MASKS)

    assert len(paths_imgs)>0
    assert len(paths_masks)>0
    assert len(paths_imgs)==len(paths_masks)

    codes = np.array(CODES)

    def label_function(filename):
           return DIR_MASKS / Path(filename.stem + filename.suffix)

    datablocks = DataBlock(blocks = (ImageBlock, MaskBlock(codes)),
                                     get_items = get_image_files,
                                     get_y = label_function,
                                     splitter = RandomSplitter(),
                                     item_tfms = RatioResize(IMG_SIZE),
                                     batch_tfms = [Normalize.from_stats(*imagenet_stats)])

    dl = datablocks.dataloaders(DIR_IMGS, bs=BATCH_SIZE, num_workers=NUM_WORKERS, device=torch.device('cuda'))
    dl.vocab = codes

    MODEL_BACKBONE = models.resnet34
    
    learn = unet_learner(dl, MODEL_BACKBONE, n_in=3, n_out=1, lr=LR, loss_func=BCEWithLogitsLossFlat(), metrics=[accuracy, dice, IoU])
    lrs = slice(LR/400, LR/4)
    learn.unfreeze()
   
   
    mlflow.fastai.autolog()

    # Start MLflow session
    with mlflow.start_run() as run: 
        learn.fit_one_cycle(EPOCHS, lrs, cbs=[ShowGraphCallback(), CSVLogger()])

    with open(learner, 'wb') as model_file:
        learn.save(model_file)

train_op = create_component_from_func(
    func=train_unet,
    packages_to_install=['fastai', 'pydantic==1.10.9', 'mlflow'],
    base_image='ultralytics/ultralytics')

show_prediction_op = create_component_from_func(
    func=show_prediction,
    packages_to_install=['fastai'],
    base_image='ultralytics/ultralytics')
  
   
@dsl.pipeline(name=PL_NAME)   
def fastai(
    pvc_name: str = 'brain-mri-volume', 
    pvc_id: str = 'pvc-c5114187-c911-4263-8caf-30415bd0ad79',
    DIR_IMGS: str = '/usr/volume/yolo/datasets/yolo_mri_brain/train/images',
    DIR_MASKS: str= '/usr/volume/yolo/datasets/yolo_mri_brain/train/images',
    CODES: list = ['background', 'tumor'], 
    IMG_SIZE: int = 256, 
    BATCH_SIZE: int = 32, 
    NUM_WORKERS: int = 0,
    LR: float = 0.001,
    EPOCHS: int = 3
):  
    RAW_VOLUME_MOUNT = mount_pvc(pvc_name=pvc_name,
                                 volume_name=pvc_id,
                                 volume_mount_path='/usr/volume/') 
     
    learner = train_op(DIR_IMGS, DIR_MASKS, CODES, 
                         IMG_SIZE, BATCH_SIZE, 
                         NUM_WORKERS, LR, 
                         EPOCHS).apply(RAW_VOLUME_MOUNT).set_gpu_limit(1)
   
    visualize_pred = show_prediction_op(learner.outputs['learner'], DIR_IMGS,DIR_MASKS,CODES, 
                         IMG_SIZE, BATCH_SIZE, 
                         NUM_WORKERS, LR).apply(RAW_VOLUME_MOUNT)
    
    
    
if __name__ == '__main__':
    import sys
    sys.path.append('./helpers')
    from deploykf_helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='fastai')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=fastai, experiment_name='test')   