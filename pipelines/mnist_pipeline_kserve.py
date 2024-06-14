from kfp.components import create_component_from_func, OutputPath, InputPath
from kfp.dsl import pipeline
from typing import NamedTuple

PL_NAME = 'Mnist_Mlflow_pipeline_example'

'''
Load and preprocess mnist dataset
'''
def load_dataset(mnist_data: OutputPath()):
    import tensorflow as tf
    import numpy as np
    import pickle 
     
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train=x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test=x_test/255.0
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
   
    data = (x_train, y_train), (x_test, y_test)
    
    with open(mnist_data, 'wb') as dataset:
        pickle.dump(data, dataset)
    
'''
Tensorflow model definition
'''   
def define_model(mnist_model: OutputPath(),
                 num_classes: int = 10
                 ):
    import tensorflow as tf
    import pickle
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
    
    with open(mnist_model, 'wb') as model_file:
        pickle.dump(model, model_file)

'''
Tensorflow training with MlFlow tracking
'''
def train(trained_model: OutputPath(),
          mlpipeline_ui_metadata_path: OutputPath(),
          mnist_data: InputPath(),
          mnist_model: InputPath(),
          batch_size: int = 64,
          epochs: int = 25,
          num_classes: int = 10
          ):
    import tensorflow as tf
    import keras
    import pickle
    import mlflow
    import matplotlib.pyplot as plt
    import mpld3
    import json
    from minio import Minio
    import glob
    import os
    
    # MlFlow setup
    mlflow.tracking.set_tracking_uri('http://mlflow-server:5000')
    experiment = mlflow.set_experiment("tf_mnist_example")
    mlflow.tensorflow.autolog()
 
    # Metics plotting
    def plot_acc_and_loss(history):
        fig, ax = plt.subplots(2,1)
        ax[0].plot(history.history['loss'], color='b', label="Training Loss")
        ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
        ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

        return mpld3.fig_to_html(fig)

    # Callback function
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
          if(logs.get('acc')>0.998):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

    with open(mnist_data, 'rb') as file:
        data = pickle.load(file)
    with open(mnist_model, 'rb') as file:
        model = pickle.load(file)
    
    (x_train, y_train), (x_test, y_test) = data
    callbacks = myCallback()   
    
    # Training
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              callbacks=[callbacks])
    
    html_output = plot_acc_and_loss(history)
    
    # Plotting metrics       
    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': html_output
        }]
    }

    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)
    
    # save model as artifact
    with open(trained_model, 'wb') as model_file:
        pickle.dump(model, model_file)
   
    # connect with minio object storage 
    minio_client = Minio(
        "10.43.245.250:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"

    # save the model in a directory for the Minio client to upload it
    keras.models.save_model(model,"/tmp/mnist")
    
    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") 
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  
                minio_client.fput_object(bucket_name, remote_path, local_file)

    # upload model to minio
    upload_local_directory_to_minio("/tmp/mnist",minio_bucket,"models/mnist/1/") # 1 for version 1
     
    

'''
Model evaluation
'''
def evaluate(mlpipeline_metrics: OutputPath(),
             mnist_data: InputPath(),
             trained_model: InputPath()):
    import pickle
    import json
    
    with open(mnist_data, 'rb') as file:
        data = pickle.load(file)
        
    with open(trained_model, 'rb') as file:
        model = pickle.load(file)

    (x_train, y_train), (x_test, y_test) = data
    
    test_loss, test_acc = model.evaluate(x_test, y_test) 

    # Exports two sample metrics for dashboard vizualization
    metrics = {
      'metrics': [{
          'name': 'Accuracy',
          'numberValue':  test_acc,
        },{
          'name': 'Loss',
          'numberValue':  test_loss,
        }]}
     
    with open(mlpipeline_metrics, 'w') as metadata_file:
        json.dump(metrics, metadata_file)
    


'''
Prediction on test dataset and visualization of confusion matrix
'''
def predict(mnist_data: InputPath(),
            trained_model: InputPath(),
            mlpipeline_ui_metadata_path: OutputPath()):
    
    import pickle
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import tensorflow as tf
    import pandas as pd
    import json
    
    with open(mnist_data, 'rb') as file:
        data = pickle.load(file)
   
    (x_train, y_train), (x_test, y_test) = data
    
    with open(trained_model, 'rb') as file:
        model = pickle.load(file)

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_test,axis = 1)
    
    # generate confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=Y_true,predictions=Y_pred_classes)
    confusion_matrix = confusion_matrix.numpy()
    vocab = list(np.unique(Y_true))
    data = []
    for target_index, target_row in enumerate(confusion_matrix):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_csv = df_cm.to_csv(header=False, index=False)
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                  ],
                "target_col" : "actual",
                "predicted_col" : "predicted",
                "source": cm_csv,
                "storage": "inline",
                "labels": ['0','1','2','3','4','5','6','7','8','9']
            }
        ]
    }
    
    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

'''
Create inference service with kserve
'''
def serve_model():
    """
    Create kserve instance
    """
    # For this it is necessary to create a service account for kserve and a secret with minio credentials in the same namespace.
    # See https://kserve.github.io/website/0.7/modelserving/storage/s3/s3/
    
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TFServingSpec
    import time

    namespace = utils.get_default_target_namespace()

    model_name='tf-mnist'
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=model_name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       service_account_name="kserve-minio-sa",
                                       tensorflow=(V1beta1TFServingSpec(
                                           storage_uri="s3://mlpipeline/models/mnist/"))))
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

    print("Model created")

'''
Functions to pipeline components
'''

load_dataset_op = create_component_from_func(
    func=load_dataset,
    packages_to_install=[],
    base_image='tensorflow/tensorflow:2.13.0-gpu')

define_model_op = create_component_from_func(
    func=define_model,
    packages_to_install=[],
    base_image='tensorflow/tensorflow:2.13.0-gpu')

train_op = create_component_from_func(
    func=train,
    packages_to_install=['mlflow', 'mpld3', 'minio'],
    base_image='tensorflow/tensorflow:2.13.0-gpu')

evaluate_op = create_component_from_func(
    func=evaluate,
    packages_to_install=['scikit-learn', 'pandas'],
    base_image='tensorflow/tensorflow:2.13.0-gpu')

predict_op = create_component_from_func(
    func=predict,
    packages_to_install=['scikit-learn','pandas'],
    base_image='tensorflow/tensorflow:2.13.0-gpu')

serve_model_op = create_component_from_func(
    func=serve_model,
    packages_to_install=['kserve', 'minio'],
    base_image='python:3.9')


'''
Pipeline creation
'''

@pipeline(name=PL_NAME)
def mnist_pipeline(
    batch_size: int = 64,
    epochs: int = 5,
    num_classes: int = 10,
):
    
    load_dataset_task = load_dataset_op()
    
    model_task = define_model_op(num_classes=num_classes)
    
    train_task = train_op(mnist_data=load_dataset_task.output,
                       mnist_model=model_task.output,
                       batch_size=batch_size,epochs=epochs,
                       num_classes=num_classes) \
                           .set_gpu_limit(1)

    predict_task = predict_op(mnist_data=load_dataset_task.output, trained_model=train_task.outputs['trained_model']) \
        .set_gpu_limit(1)
    eval_task = evaluate_op(mnist_data=load_dataset_task.output, trained_model=train_task.outputs['trained_model']) \
        .set_gpu_limit(1)
    serve_task = serve_model_op().after(train_task)
    #serve_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
   
   
    
if __name__ == '__main__':
    
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='mnist')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=mnist_pipeline, experiment_name='test')