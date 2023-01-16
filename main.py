from A1.task_a1 import TaskA1Evaluator
from A2.task_a2 import TaskA2Evaluator
from B1.task_b1 import TaskB1Evaluator
from B2.task_b2 import TaskB2Evaluator
import argparse
import time


def parse_argument():
    """Get the argument supplied by the user, which indicates the task to be run."""
    allowed_arguments = ["task_a1", "task_a2", "task_b1", "task_b2"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="task to be run", required=True)
    arguments = vars(parser.parse_args())  # Parse the input arguments

    # If the user has selected an allowed task, return its index.
    # Otherwise, return -5
    if arguments['task'] in allowed_arguments:
        return allowed_arguments.index(arguments['task'])
    else:
        return -5

def main():
    task = parse_argument()  # Get the index of the task selected by the user
    allowed_tasks = ['A1', 'A2', 'B1', 'B2']

    # Initialize the evaluator of the task selected by the user
    if task == 0:
        evaluator = TaskA1Evaluator()
    if task == 1:
        evaluator = TaskA2Evaluator()
    if task == 2:
        evaluator = TaskB1Evaluator()
    if task == 3:
        evaluator = TaskB2Evaluator()
    
    # If the use has selected a task that is not allowed
    if task == -5:
        print("The selected task does not exist. Please, type one of the following \
        arguments: task_a1, task_a2, task_b1, task_b2")

    # If the task selected is allowed, run it
    else:
        print(f'Running Task {allowed_tasks[task]} shortly.')
        time.sleep(5)
        evaluator.run_task() 


if __name__ == "__main__":
    main()