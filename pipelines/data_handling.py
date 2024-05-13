import kfp
import sys
from kfp.components import create_component_from_func, OutputPath, InputPath
from typing import NamedTuple
'''
Pipeline Functions
'''
# Simple multiplication of two values. Annotated to return a NamedTuple containing all input values and the output value (result).
# This is done this way to showcase handling of multiple outputs.
def multiply(value_1: float = 1.5,
             value_2: float = 1.5) -> float:#NamedTuple('outputs',[('value_1', float),('value_2', float),('result', float)]):
    
    from typing import NamedTuple
    result = value_1 * value_2 
    return float(result)#NamedTuple('outputs', [('value_1', float),('value_2', float),('result', float)])

# Writes text to output file. OutputPath() paramater annotation produces output data as a file.
# It passes the PATH of a file where the function writes output data and uploads it after execution.
def write_result_to_text_file(text_file: OutputPath(),
                              result: float = 0):
    with open(text_file, 'w') as outfile:
        outfile.write('The result is:' + str(result))

# Gets and returns content from file. 
# InputPath() parameter annotation tells the function it will receive the corresponding data as a file.
# The system will download the data and pass it to the function as PATH
def get_text_file(text_file: InputPath()) -> str:
    with open(text_file, 'r') as infile:
        data = infile.read()
    return data

'''
Functions to pipeline components
'''
# Create pipeline components from each function to import/call from the pipeline function.
multiply_op = create_component_from_func(
    func=multiply,  # choose function to create component from
    packages_to_install=[], # list packages that need to be installed additionally e.g. `packages_to_install=['numpy', 'pandas']`
    base_image='python:3.9') # choose the image you want to use

write_result_to_text_file_op = create_component_from_func(
    func=write_result_to_text_file,
    packages_to_install=[],
    base_image='python:3.9')

get_text_file_op = create_component_from_func(
    func=get_text_file,
    packages_to_install=[],
    base_image='python:3.9')


'''
Pipeline creation
'''
# Pipeline function
@kfp.dsl.pipeline(name='Example Pipeline')
def test_pipeline(
    value_1: float = 1.5,
    value_2: float = 1.5
):
    # multiply user input
    multiply_task = multiply_op(value_1, value_2)
    # create text file
    text_file_task = write_result_to_text_file_op(result=multiply_task.output)
    # get text file
    text_file_data_task = get_text_file_op(text=text_file_task.output)


if __name__ == '__main__':
    
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='ecample')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=test_pipeline, experiment_name='test')
        
#    # Compile pipeline as .yaml.
#    kfp.compiler.Compiler().compile(
#        pipeline_func=test_pipeline,
#        package_path='test_pipeline.yaml')
#