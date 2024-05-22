from kfp import dsl
from kfp.components import create_component_from_func, OutputPath, InputPath
from kfp.onprem import mount_pvc

'''
Check if gpu/cuda is available
'''
def gpu_available_check_op():
    return dsl.ContainerOp(
        name='check gpu',
        image='ultralytics/ultralytics',
        command=['sh', '-c'],
        arguments=['nvidia-smi']
    ).set_gpu_limit(2)

'''
Train defined yolo model on chosen dataset
'''
def yolo_train(mlpipeline_metrics: OutputPath(),
               model: str = 'yolov8n.pt',
               data: str = 'coco128.yaml',
               epochs: int = 5,
               batch: int = 16,
               save_path: str ='/example-volume',
               mlflow_experiment_name: str = 'yolo-example'):
    import os
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow-server:5000'
    os.environ['MLFLOW_REGISTRY_URI'] = 'http://mlflow-server:5000'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
    import mlflow
    import json
    from ultralytics import YOLO
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, workers=0, project=save_path+'/yolo', device=[0, 1], batch=batch)

    metrics = model.val(workers=0, project=save_path+'/yolo')
    
    results = metrics.results_dict
    
    # Exports metrics for visualization in Kubeflow UI:
    metrics = {
      'metrics': [{
          'name': 'Precision',
          'numberValue':  results['metrics/precision(B)'],
        },{
          'name': 'Recall',
          'numberValue':  results['metrics/recall(B)'],
        },{
            'name': 'MaP50',
            'numberValue':  results['metrics/mAP50(B)'],
        }]}
     
    with open(mlpipeline_metrics, 'w') as metadata_file:
        json.dump(metrics, metadata_file)
 
'''
Predict with best checkpoint from training on image_path/url
''' 
def yolo_predict(prediction: OutputPath(),
                 predict_data: str = '',
                 save_path: str ='/example-volume/',
                 chkpt_path:str = '/train/weights/best.pt' # checkpoint path relative to save path
                 ):
    from ultralytics import YOLO
    import pickle
   
    # predict on single image 
    model = YOLO(save_path+ '/yolo' + chkpt_path)
    pred = model(predict_data, project=save_path)  # predict on an image
    
    with open(prediction, 'wb') as outfile:
        pickle.dump(pred, outfile)


''' 
draws predicted bounding box on image and vizualizes it
''' 
def draw_bbox(prediction_image_path: OutputPath('png'),
              prediction: InputPath()
              ):   
    import cv2
    import pickle
    import numpy as np
    
    def draw_bboxes_on_img(pred):
        img_array = pred.orig_img
        img = img_array.astype(np.uint8)
        
        dh,dw, _ = img.shape
        for idx, box in enumerate(pred.boxes.xywhn):
            x,y,w,h = box.tolist()

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2) 
            name = pred.names[int(pred.boxes.cls.tolist()[idx])]
            cv2.putText(img, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        return img
    
    with open(prediction, 'rb') as infile:
        pred = pickle.load(infile)
   
    img = draw_bboxes_on_img(pred[0])
    
    with open(prediction_image_path, 'wb') as outfile:
        pickle.dump(img, outfile)


''' 
Visualizes image from path in 'visualize tab' 
''' 
def visualize_image(image_path: InputPath(),
                    mlpipeline_ui_metadata_path: OutputPath()):
    import json
    import matplotlib.pyplot as plt
    import mpld3
    import pickle
    import numpy as np
    
    with open(image_path, 'rb') as infile:
        img = pickle.load(infile)
   
    img = img.astype(np.uint8)
    fig = plt.figure()
    plt.imshow(img)
    
    html_plot = mpld3.fig_to_html(fig)

    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': html_plot
        }]
    }

    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

   
''' 
Functions to pipeline components
''' 
train_op = create_component_from_func(
    func=yolo_train,
    packages_to_install=['mlflow','psutil','pynvml'],
    base_image='ultralytics/ultralytics')

predict_op = create_component_from_func(
    func=yolo_predict,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

visualize_image_op = create_component_from_func(
    func=visualize_image,
    packages_to_install=['matplotlib', 'mpld3'],
    base_image='python:3.9')

draw_bbox_op = create_component_from_func(
    func=draw_bbox,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

'''
Pipeline creation
'''
@dsl.pipeline(name='yolo-example')   
def yolo_object_detection(
     model: str = 'yolov8n.pt',
     data: str = 'coco128.yaml',
     batch:int = 32,
     epochs: int = 15,
     predict_data: str = 'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/image_data/zebra.jpg',
     pvc_name: str = 'kfpv1-workshop-volume',
     pvc_id: str = 'pvc-9d4173e6-0908-41bf-8bba-e2bbbecaa452',
     pvc_mount_path: str = '/example-volume',
     mlflow_experiment_name: str = 'yolo-example'
 ):   
     RAW_VOLUME_MOUNT = mount_pvc(pvc_name=pvc_name,
                             volume_name=pvc_id,
                             volume_mount_path=pvc_mount_path)
    
     # GPU Check
     gpu_check = gpu_available_check_op()
     
     # Yolo training task on coco dataset
     yolo_train_task = train_op(model=model, data=data, batch=batch, epochs=epochs,save_path=pvc_mount_path, mlflow_experiment_name=mlflow_experiment_name) \
         .after(gpu_check) \
         .apply(RAW_VOLUME_MOUNT) \
         .set_gpu_limit(2)
 
     # Yolo prediction on input image
     yolo_predict_task = predict_op(predict_data=predict_data, save_path = pvc_mount_path) \
         .after(yolo_train_task) \
         .apply(RAW_VOLUME_MOUNT) \
         
     # visualize prediction result
     draw_boundingbox_task = draw_bbox_op(prediction=yolo_predict_task.output)
     
     visualize_image_task = visualize_image_op(image=draw_boundingbox_task.output)



if __name__ == '__main__':
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='yolo')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=yolo_object_detection, experiment_name='test')   
    