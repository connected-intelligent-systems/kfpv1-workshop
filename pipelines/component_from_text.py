from kfp import components
from kfp import dsl

multiply_comp = components.load_component_from_text(
    '''
name: Multiply
description: |
    Multiplication
inputs:
- name: value1
  type: Integer
- name: value2
  type: Integer
outputs:
- name: result
  type: Integer
implementation:
  container:
    image: alpine:latest
    command:
    - sh
    - -c
    - |
      set -e -x
      mkdir -p $(dirname "$2")
      echo "$(($0*$1))" > /tmp/result
      cp /tmp/result "$2"
    - {inputValue: value1}
    - {inputValue: value2}
    - {outputPath: result}
'''
)

'''
Pipeline
'''
@dsl.pipeline(name='add-pipeline')
def load_from_text(
    a: int = 2,
    b: int = 5,
):
    multiply = multiply_comp(a, 3)
    multiply2 = multiply_comp(multiply.outputs['result'], b)
   
  
  
   
if __name__ == '__main__':
    from kfpv1helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='load_from_text')
    #helper.upload_pipeline(pipeline_function=load_from_text)
    helper.create_run(pipeline_function=load_from_text, experiment_name='test')   