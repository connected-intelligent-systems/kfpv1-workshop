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
               epochs: int = 5,
               save_path: str ='/usr/example-pipeline-volume/yolo',
               mount_path: str = '/usr/share',
               imgsz: int = 640,
               batch_size: int = 8,
               mosaic: float = 0.3,
               scale: float = 0.5,
               patience: int = 100,
               lr0: float = 0.01,
               lrf: float = 0.01,
               optimizer: str = 'auto',
               warmup_epochs: float = 3.0,
               mlflow_experiment_name: str = 'ultralytics yolo')  -> NamedTuple('Outputs', [('onnx', str), ('model_path', str)]): # type: ignore
    
    import os
    from pathlib import Path
    
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow-server:5000'
    os.environ['MLFLOW_REGISTRY_URI'] = 'http://mlflow-server:5000'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
    import mlflow
    
    from ultralytics import YOLO   
    new_data = os.path.join(mount_path, data)
    new_save_path = os.path.join(mount_path, save_path)

    model = YOLO(model)
    model.train(workers=0, data=new_data, epochs=epochs, project=new_save_path, batch=batch_size, imgsz=imgsz, mosaic=mosaic, scale=scale, patience=patience, lr0=lr0, lrf=lrf, optimizer=optimizer, warmup_epochs=warmup_epochs)
   
       # exports model to onnx
    export = model.export(format='onnx', dynamic=True)

    # Define paths for serving
    triton_repo_path = Path(new_save_path) / 'triton_repo'
    triton_model_path = triton_repo_path / 'brain-mri'

    # Create directories
    (triton_model_path / '1').mkdir(parents=True, exist_ok=True)
    # Move ONNX model to Triton Model path
    Path(export).rename(triton_model_path / '1' / 'model.onnx')
    
    # Create config file
    (triton_model_path / 'config.pbtxt').touch()
    # write to config file
    with open(triton_model_path / 'config.pbtxt', 'w') as f:
        f.write('''name: "brain-mri"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
{
  name: "images"
  data_type: TYPE_FP32
  dims: [-1,3,-1,-1]
}
]
output [
{
  name: "output0"
  data_type: TYPE_FP32
  dims: [-1,-1,-1]
}
]''')
    
    values = {"onnx": str(os.path.join(save_path, 'triton_repo')), "model_path": str(model.trainer.best)}
    
    from collections import namedtuple
    output = namedtuple(
      'Outputs',
      ['onnx', 'model_path'])
    
    return output((os.path.join(save_path, 'triton_repo')),str(model.trainer.best)) 
     
    
 
def yolo_val(chkpt_path: str, save_path: str, mount_path: str) -> NamedTuple('outputs', [('metrics', str), ('high_map_score', bool)]):
    
    from ultralytics import YOLO
    import os

    save_path = os.path.join(mount_path, save_path)

    # Load a model
    model = YOLO(chkpt_path)  # load a custom model

    # Validate the model
    metrics = model.val(workers=0, project=save_path)

    if metrics.results_dict['metrics/mAP50(B)'] > 0.8:
        high_map_score = True
    else:
        high_map_score = False

    return [str(metrics.save_dir), high_map_score] 
    
'''
Visualizes image from path in 'visualize tab'
'''
def visualize_validation_batches(image_path: str,
                                 truth_batch_name: str,
                                 pred_batch_name: str,
                                 )-> NamedTuple('VisualizationOutput', [('mlpipeline_ui_metadata', 'UI_metadata')]):
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
        
    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': html,
        }]
    }
    from collections import namedtuple
    visualization_output = namedtuple('VisualizationOutput', [
        'mlpipeline_ui_metadata'])
    
    return visualization_output(json.dumps(metadata))

'''
Create a kserve service
'''
def kserve_scv(model: str,
               model_name: str = "brain-mri", 
               kserve_version: str ='v1beta1',
               pvc_name: str = 'kfpv1-workshop-volume'):
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1ONNXRuntimeSpec
    import time
    
    model_path = model 

    model = 'pvc://%s' % pvc_name + '/' + model_path 
    print(model)
    namespace = utils.get_default_target_namespace()
   
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=model_name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false', 'nginx.ingress.kubernetes.io/proxy-body-size': '900m'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                   onnx=(V1beta1ONNXRuntimeSpec(
                                       storage_uri=model, runtime_version='24.01-py3'))))
    )

    KServe = KServeClient()
  
    # Create or update the inference service
    try:
        KServe.delete(model_name)
        time.sleep(30)
        print("Model deleted")
    except:
        print("Service does not exist yet!")

    KServe.create(isvc, watch=True)   

'''
KFP component creation
'''
train_op = create_component_from_func(
    func=yolo_train,
    packages_to_install=['mlflow'],
    base_image='ultralytics/ultralytics')

predict_op = create_component_from_func(
    func=yolo_val,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

vis_op = create_component_from_func(
    func=visualize_validation_batches,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

serve_op = create_component_from_func(
    func=kserve_scv,
    packages_to_install=['kserve'],
    base_image='ultralytics/ultralytics')
   
@dsl.pipeline(name=PL_NAME)   
def yolo_object_detection(
    model: str = 'yolov8n-seg.yaml',
    data: str = 'kfpv1-workshop/brain-mri/notebooks_data_preparation/yolo/datasets/mri_brain.yaml',
    epochs: int = 10,
    num_gpus: int = 1,
    pvc_id: str = 'pvc-9d4173e6-0908-41bf-8bba-e2bbbecaa452',
    pvc_name: str = 'kfpv1-workshop-volume',
    save_path: str = 'kfpv1-workshop/brain-mri/notebooks_data_preparation/yolo',
    image_size: int = 256,
    batch_size: int = 512,
    mosaic: float = 0.3,
    scale: float = 0.5,
    patience: int = 50,
    initial_learning_rate: float = 0.01,
    final_learning_rate: float = 0.01, 
    optimizer: str = 'SGD',
    warmup_epochs: float = 3.0,
    mlflow_experiment_name: str = 'yolo-brain-mri',
    mount_path: str = '/usr/share'
    ):  
    RAW_VOLUME_MOUNT = mount_pvc(pvc_name=pvc_name,
                                 volume_name=pvc_id,
                                 volume_mount_path=mount_path) 
     
    # Yolo training task on coco dataset
    yolo_train_task = train_op(model=model,data=data,epochs=epochs,save_path=save_path,mount_path=mount_path,imgsz=image_size,batch_size=batch_size,mosaic=mosaic,scale=scale,patience=patience,lr0=initial_learning_rate,lrf=final_learning_rate,optimizer=optimizer,warmup_epochs=warmup_epochs,mlflow_experiment_name=mlflow_experiment_name
                                 ).apply(RAW_VOLUME_MOUNT).set_gpu_limit(1)
    
    yolo_validation = predict_op(chkpt_path=yolo_train_task.outputs["model_path"], save_path=save_path, mount_path=mount_path).apply(RAW_VOLUME_MOUNT)

    with dsl.Condition(yolo_validation.outputs['high_map_score'] == True):
        serving_task = serve_op(model=yolo_train_task.outputs["onnx"], pvc_name=pvc_name).apply(RAW_VOLUME_MOUNT)
        serving_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    # visualize some validation batches
    visualize_val = vis_op(image_path=yolo_validation.outputs['metrics'], truth_batch_name='val_batch0_labels.jpg', pred_batch_name='val_batch0_pred.jpg').apply(RAW_VOLUME_MOUNT) # type: ignore


if __name__ == '__main__':
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='yolo-brain-mri')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=yolo_object_detection, experiment_name='test')   
    