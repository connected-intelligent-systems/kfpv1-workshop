
from kfp.components import create_component_from_func, OutputPath, InputPath
from kfp.dsl import pipeline
from kfp.onprem import mount_pvc
'''
Pipeline Functions
'''
# Downlaod data from url to a mounted volume
def get_example_data_to_volume(url: str = 'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/tabular_data/some_metrics.csv',
                               save_path: str = '/usr/share/example-pipeline-volume/zebra.jpg'
                               ):
    import requests
    
    r = requests.get(url, allow_redirects=True)
    
    if r.status_code == 200:
        open(save_path,'wb').write(r.content)
    else: 
        raise Exception('Could not get file from url')


# Get Image from Volume and visualize in kubeflow
def display_image_from_path(mlpipeline_ui_metadata_path: OutputPath(),
                        img_file: str = '/usr/share/example-pipeline-volume/zebra.jpg'
                        ):
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import mpld3
    import json
    
    img = np.asarray(Image.open(img_file))
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

    return img

# Plot csv-file in interactive visualization
def plot_csv_from_path(mlpipeline_ui_metadata_path: OutputPath(),
             csv_file: str = '/usr/share/example-pipeline-volume/data.csv'
            ):
    import pandas as pd
    import mpld3
    import json
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.1)
    
    for column in df.columns:
        l, = ax.plot(df[column].index, df[column].values, label=column)
     
    handles, labels = ax.get_legend_handles_labels()
   
    interactive_legend = mpld3.plugins.InteractiveLegendPlugin(zip(handles,
                                                             ax.collections),
                                                         labels,
                                                         alpha_unsel=0.5,
                                                         alpha_over=1.5, 
                                                         start_visible=True)
    mpld3.plugins.connect(fig, interactive_legend)
   
    ax.set_xlabel('epochs')
    ax.set_ylabel('metric')
    ax.set_title('Metric visualization', size=15)

    html_output = mpld3.fig_to_html(fig)

    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': html_output
        }]
    }

    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

'''
Functions to pipeline components
'''

get_example_data_to_volume_op = create_component_from_func(
    func=get_example_data_to_volume,
    packages_to_install=['requests'],
    base_image='python:3.9')

plot_csv_from_path_op = create_component_from_func(
    func=plot_csv_from_path,
    packages_to_install=['matplotlib','mpld3','pandas'],
    base_image='python:3.9')

display_image_from_path_op = create_component_from_func(
    func=display_image_from_path,
    packages_to_install=['numpy','pillow','matplotlib','mpld3'],
    base_image='python:3.9')

'''
Pipeline creation
'''

@pipeline(name='data_viz')   
def data_visualization(
    csv_url: str = 'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/tabular_data/some_metrics.csv',
    img_url: str =  'https://gitlab.com/sebastian.hocke96/example_files/-/raw/main/image_data/zebra.jpg',
    img_save_path: str = '/usr/share/example-pipeline-volume/zebra.jpg',
    csv_save_path: str = '/usr/share/example-pipeline-volume/data.csv' 
    ):   
    
    RAW_VOLUME_MOUNT = mount_pvc(pvc_name= 'example-volume',
                             volume_name='pvc-f39af9b8-9114-41c4-b9a7-bbfe843629f0',
                             volume_mount_path='/usr/share/example-pipeline-volume')
    
    
    # save csv data from url to RAW_VOLUME_MOUNT
    csv_to_vol_task = get_example_data_to_volume_op(csv_url, csv_save_path)
    csv_to_vol_task.apply(RAW_VOLUME_MOUNT)

    # read csv file from path and plot table
    csv_plot_task = plot_csv_from_path_op(csv_file=csv_save_path)
    csv_plot_task.apply(RAW_VOLUME_MOUNT)
    csv_plot_task.after(csv_to_vol_task)

    # get image from path
    img_to_vol_task = get_example_data_to_volume_op(img_url, img_save_path)
    img_to_vol_task.apply(RAW_VOLUME_MOUNT)

    # save image from url to RAW_VOLUME_MOUNT
    display_image_task = display_image_from_path_op(img_file=img_save_path)
    display_image_task.apply(RAW_VOLUME_MOUNT)
    display_image_task.after(img_to_vol_task)

'''
Pipeline creation
'''
if __name__ == '__main__':
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='data_viz')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=data_visualization, experiment_name='test')