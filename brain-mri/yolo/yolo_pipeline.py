from kfp import dsl
from kfp.components import create_component_from_func, OutputPath, InputPath
from typing import NamedTuple
from kfp.onprem import mount_pvc

PL_NAME = 'ultralytics_yolo'

'''
Train defined yolo model on chosen dataset
'''
def yolo_train(model: str = 'yolov8n-seg.pt',
               data: str = 'coco128-seg.yaml',
               epochs: int = 3,
               save_path: str ='/usr/example-pipeline-volume/yolo',
               imgsz: int = 640,
               batch_size: int = 8,
               mosaic: float = 0.3,
               scale: float = 0.5,
               patience: int = 100,
               lr0: float = 0.01,
               lrf: float = 0.01,
               optimizer: str = 'auto',
               warmup_epochs: float = 3.0,
               mlflow_experiment_name: str = 'ultralytics yolo') -> str:
    import os
    # https://github.com/ultralytics/ultralytics/commit/1ae7f8439465c4bbba9eb939cb138903373ee1e8
    # Need to wait for this to release for Mlflow_tracking, should be 15.August 2023 
    # 
    # os.environ['MLFLOW_TRACKING_URI'] = 'http://mlf-mlflow.kubeflow.svc.cluster.local:5000'
    # os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
    # import mlflow
    
    from ultralytics import YOLO   
    model = YOLO(model)
    model.train(workers=0, data=data, epochs=epochs, project=save_path, batch=batch_size, imgsz=imgsz, mosaic=mosaic, scale=scale, patience=patience, lr0=lr0, lrf=lrf, optimizer=optimizer, warmup_epochs=warmup_epochs)
    
    return str(model.trainer.best) # Returns path of the best checkpoint
 
 
def yolo_val(chkpt_path: str, save_path: str) -> str:
    
    from ultralytics import YOLO

    # Load a model
    model = YOLO(chkpt_path)  # load a custom model

    # Validate the model
    metrics = model.val(workers=0, project=save_path)
    
    return str(metrics.save_dir) 
    
'''
Visualizes image from path in 'visualize tab'
'''
def visualize_validation_batches(image_path: str,
                                 truth_batch_name: str,
                                 pred_batch_name: str,
                                 image_visualization: OutputPath()):
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
  
    val_batch = img_to_base64(os.path.join(image_path, truth_batch_name))
    pred_batch = img_to_base64(os.path.join(image_path, pred_batch_name))
   
    html = '<!DOCTYPE html><html><body><h2>Ground truth</h2><img src='+val_batch+' alt="ground truth"></body><body><h2>Prediction</h2><img src='+pred_batch+' alt="prediction"></body></html>' 
    
    with open(image_visualization, 'w') as metadata_file:
        json.dump(html, metadata_file)

'''
KFP component creation
'''
train_op = create_component_from_func(
    func=yolo_train,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

predict_op = create_component_from_func(
    func=yolo_val,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

vis_op = create_component_from_func(
    func=visualize_validation_batches,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

   
@dsl.pipeline(name=PL_NAME)   
def yolo_object_detection(
    model: str = 'yolov8n-seg.yaml',
    data: str = '/usr/volume/yolo/datasets/mri_brain.yaml',
    epochs: int = 5,
    num_gpus: int = 1,
    pvc_id: str = 'pvc-c5114187-c911-4263-8caf-30415bd0ad79',
    pvc_name: str = 'brain-mri-volume',
    save_path: str = '/usr/volume/yolo',
    image_size: int = 256,
    batch_size: int = 512,
    mosaic: float = 0.3,
    scale: float = 0.5,
    patience: int = 100,
    initial_learning_rate: float = 0.01,
    final_learning_rate: float = 0.01, 
    optimizer: str = 'SGD',
    warmup_epochs: float = 3.0,
    mlflow_experiment_name: str = 'yolo-brain-mri'
    ):  
    RAW_VOLUME_MOUNT = mount_pvc(pvc_name=pvc_name,
                                 volume_name=pvc_id,
                                 volume_mount_path='/usr/volume/') 
     
    # Yolo training task on coco dataset
    yolo_train_task = train_op(model=model,data=data,epochs=epochs,save_path=save_path,imgsz=image_size,batch_size=batch_size,mosaic=mosaic,scale=scale,patience=patience,lr0=initial_learning_rate,lrf=final_learning_rate,optimizer=optimizer,warmup_epochs=warmup_epochs,mlflow_experiment_name=mlflow_experiment_name
                                 ).apply(RAW_VOLUME_MOUNT).set_gpu_limit(1)
    
    yolo_validation = predict_op(chkpt_path=yolo_train_task.output, save_path=save_path).apply(RAW_VOLUME_MOUNT)
    
    # visualize some validation batches
    visualize_val = vis_op(image_path=yolo_validation.output, truth_batch_name='val_batch0_labels.jpg', pred_batch_name='val_batch0_pred.jpg').apply(RAW_VOLUME_MOUNT) # type: ignore


if __name__ == '__main__':
    import sys
    sys.path.append('./helpers')
    from deploykf_helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='yolo')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=yolo_object_detection, experiment_name='test')   
    