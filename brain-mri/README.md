Dataset link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

## Prepare Dataset
1. Create a jupyter notebook server in the Kubeflow UI with at least 10gb workspace volume and connect to it.
2. Copy the content of **~/separate_examples/brain_tumor_detection/notebooks_data_preparation** into the root of your created JupyterLab UI (drag&drop).
3. Use the download_and_extract_dataset.ipynb. For this you have to generate a new link with your own kaggle account of the dataset linked in the comment. (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
4. Use the prepare_data.ipynb

## For yolo training
5. Go to the pipelines tab in the Kubeflow UI and search for the **ultralytics_yolo** - pipeline
6. Run the pipeline, but change the name of the volume to the workspace volume of your created notebook
7. Tracking can be done by creating a tensorboard in the Kubeflow UI attached to the volume of your created notebook.
8. Tracking is also available in MLflow. By default the experiment is named *yolo-brain-mri*

## For the fastai-unet example
5. Go to the pipelines tab in the Kubeflow UI and search for the **ultralytics_yolo** - pipeline
6. Run the pipeline, but change the name of the volume to the workspace volume of your created notebook. 
7. The training can be tracked in the MLFlow tab. By default the experiment name is *fastai-brain-mri*

## For the pytorch-unet example
5. Go to the pipelines tab in the Kubeflow UI and search for the **ultralytics_yolo** - pipeline
6. Run the pipeline, but change the name of the volume to the workspace volume of your created notebook. 
