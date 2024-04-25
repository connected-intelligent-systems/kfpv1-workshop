from kfp import dsl
from kfp.components import create_component_from_func, OutputPath, InputPath
from typing import NamedTuple
from kfp.onprem import mount_pvc

# Check if gpu/cuda is available
def gpu_available_check_op():
    return dsl.ContainerOp(
        name='check gpu',
        image='ultralytics/ultralytics',
        command=['sh', '-c'],
        arguments=['nvidia-smi']
    ).set_gpu_limit(2)


# Train defined yolo model on chosen dataset
def yolo_train(model: str = 'yolov8n.pt',
               data: str = 'coco128.yaml',
               epochs: int = 3,
               batch: int = 16,
               save_path: str ='/usr/share/example-pipeline-volume/yolo'):
    
    from ultralytics import YOLO
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, workers=0, project=save_path, device=[0, 1], batch=batch)
 
    
# Predict with best checkpoint from training on image_path/url
def yolo_predict(prediction: OutputPath(),
                 predict_data: str = '',
                 save_path: str ='/usr/share/example-pipeline-volume/yolo',
                 chkpt_path:str = '/train/weights/best.pt' # checkpoint path relative to save path
                 ):
    from ultralytics import YOLO
    import pickle
    
    model = YOLO(save_path + chkpt_path)
    pred = model(predict_data, project=save_path)  # predict on an image
    
    with open(prediction, 'wb') as outfile:
        pickle.dump(pred, outfile)


# draws predicted bounding box on image
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


# Visualizes image from path in 'visualize tab' 
def visualize_image(image_path: InputPath(),
                    mlpipeline_ui_metadata_path: OutputPath()):
    import json
    import matplotlib.pyplot as plt
    import mpld3
    import pickle
    
    with open(image_path, 'rb') as infile:
        img = pickle.load(infile)
    
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

   
# Opens tensorboard inside the visualization tab. Only works with single tensorboard-files, not directories (??)
def tensorboard_visualisation(mlpipeline_ui_metadata_path: OutputPath(),
                              tb_log: str = 'log_file'):
    import json
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': tb_log 
        }]
    }   
    
    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)
        
 

train_op = create_component_from_func(
    func=yolo_train,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

predict_op = create_component_from_func(
    func=yolo_predict,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')

tensorboard_visualization_op = create_component_from_func(
    func=tensorboard_visualisation,
    packages_to_install=[],
    base_image='python:3.9')

visualize_image_op = create_component_from_func(
    func=visualize_image,
    packages_to_install=['matplotlib', 'mpld3'],
    base_image='python:3.9')

draw_bbox_op = create_component_from_func(
    func=draw_bbox,
    packages_to_install=[],
    base_image='ultralytics/ultralytics')


@dsl.pipeline(name='yolo-example')   
def yolo_object_detection(
     model: str = 'yolov8n.pt',
     data: str = 'coco128.yaml',
     batch:int = 32,
     epochs: int = 60,
     predict_data: str = 'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/image_data/zebra.jpg',
 ):   
     RAW_VOLUME_MOUNT = mount_pvc(pvc_name= 'pvc-24243d72-e5a8-488c-aab3-3a17f979e6ed',
                             volume_name='example-volume',
                             volume_mount_path='/usr/share/example-pipeline-volume')
    
     # GPU Check
     gpu_check = gpu_available_check_op()
     
     # Yolo training task on coco dataset
     yolo_train_task = train_op(model, data, batch, epochs,save_path ='/usr/share/example-pipeline-volume/yolo') \
         .after(gpu_check) \
         .apply(RAW_VOLUME_MOUNT) \
         .set_gpu_limit(2)
 
     # Yolo prediction on input image
     yolo_predict_task = predict_op(predict_data=predict_data) \
         .after(yolo_train_task) \
         .apply(RAW_VOLUME_MOUNT) \
         .set_gpu_limit(2) \
         
     # get tensorboard example file
     init_tensorboard_task = tensorboard_visualization_op(tb_log='/usr/share/example-pipeline-volume/example-volume/yolo/train') \
         .after(yolo_train_task) \
         .apply(RAW_VOLUME_MOUNT) \
         
     # visualize prediction result
     draw_boundingbox_task = draw_bbox_op(prediction=yolo_predict_task.output)
     visualize_image_task = visualize_image_op(image=draw_boundingbox_task.output)



if __name__ == '__main__':
    import sys
    sys.path.append('./helpers')
    from deploykf_helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='yolo')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=yolo_object_detection, experiment_name='test')   
    
        













    # model = 'yolov8n.pt'
    # epochs = 2
    # data = 'coco128.yaml'
    # predict_data = 'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/image_data/zebra.jpg'
   
    # yolo_train(model, data, epochs) 
    # pred = yolo_predict(model, predict_data)
    # 