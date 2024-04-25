from kfp.components import create_component_from_func,create_graph_component_from_pipeline_func , OutputPath, InputPath
from kfp import dsl
from kfp.dsl import pipeline

'''
Pipeline Functions
'''
# Returns True if the input is a prime number
def check_if_prime(number: int = 0) -> bool:
    for i in range(2,number):
            if (number%i) == 0:
                return False
    return True

# substract one from input parameter. Done this way, because we can't perform mathematical operations on Pipeline Parameters in the Pipeline Function.
def sub_one(number: int = 12,
            one: int = 1) -> int:
    result = number - one
    return result

# Simple print to log component
def cs_print(number: int = '23') -> int:
    print(number)
    return number

# Exit task to demonstrate exit_handling
def exit_task():
    print('This is the exit task')

# Graph component decorator for recursively called function
@dsl.graph_component
def find_next_prime(number: int = 13):
    is_prime = check_if_prime_op(number)
    # set max_cache_staleness to 0 to prevent infinite loop due to caching
    is_prime.execution_options.caching_strategy.max_cache_staleness = "P0D"
    with dsl.Condition(is_prime.output == False):
        sub_one = sub_one_op(number=number).after(is_prime)
        find_next_prime(number=sub_one.outputs['output'])
        print_result = cs_print(number)


'''
Functions to pipeline components
'''
check_if_prime_op = create_component_from_func(
    func=check_if_prime,
    packages_to_install=[],
    base_image='python:3.9')

print_op = create_component_from_func(
    func=cs_print,
    packages_to_install=[],
    base_image='python:3.9')

sub_one_op = create_component_from_func(
    func=sub_one,
    packages_to_install=[],
    base_image='python:3.9')

exit_op = create_component_from_func(
    func=exit_task,
    packages_to_install=[],
    base_image='python:3.9')

'''
Pipeline creation
'''
@pipeline(name='recursion')
def recursion_conditions_and_exit_handlers(number: int = 20):

    exit_task = exit_op()
    # ExitHandler executes exit_task after pipeline completion no matter which result
    with dsl.ExitHandler(exit_task):
        # First check if number is prime
        first_try = check_if_prime_op(number)
        first_try.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # if number is not prime (Condition)
        with dsl.Condition(first_try.output == False):
            # find the next lower prime number
            find_next_prime_task = find_next_prime(number)


if __name__ == '__main__':
    import sys

    sys.path.append('./helpers')
    from deploykf_helper import kfphelpers
    
    helper = kfphelpers(namespace='workshop', pl_name='recursion')
    #helper.upload_pipeline(pipeline_function=yolo_object_detection)
    helper.create_run(pipeline_function=recursion_conditions_and_exit_handlers, experiment_name='test')