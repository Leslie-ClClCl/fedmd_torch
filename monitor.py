import os
import subprocess
import threading
import time

import pynvml

pynvml.nvmlInit()
DEVICE_NUM = 2
TASKS_AT_MOST = 2

envs = os.environ['PATH']
envs += ':/home/lichenglong/miniconda3/envs/fedmd_torch/bin/'


def getGPUmem(idx):
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / (1024 * 1024)


class RunTasks_thread(threading.Thread):
    def __init__(self, task_idx, task):
        threading.Thread.__init__(self)
        self.task_idx = task_idx
        self.task = task

    def run(self):
        time.sleep(self.task_idx * 10)
        print("Task {} start".format(self.task_idx))
        subprocess.call(self.task, env={'PATH': envs})
        print("Task {} finish".format(self.task_idx))


def getActiveTaskNum(pool):
    ctr = 0
    for idx, item in enumerate(pool):
        if item.is_alive():
            ctr += 1
        # TODO remove thread that isn't alive
    return ctr


if __name__ == '__main__':
    tasks = [
        [r'python', 'cifar_balanced.py', '-models', 'resnet50', '-private_classes', '60', '-result_saved_path',
         'results'],
        # ['python', 'cifar_balanced.py', '-models', 'resnet50', '-private_classes', '30', '-result_saved_path', 'results'],
        [r'python', 'cifar_balanced.py', '-models', 'resnet50', '-private_classes', '18', '-result_saved_path',
         'results'],
        [r'python', 'cifar_balanced.py', '-models', 'resnet50', '-private_classes', '6', '-result_saved_path',
         'results'],
    ]
    todo_idx = 0
    task_doing_pool = []
    while todo_idx < len(tasks):
        for idx in range(DEVICE_NUM):
            freeMem = getGPUmem(idx)
            if freeMem > 10000 and getActiveTaskNum(task_doing_pool) < TASKS_AT_MOST:
                print(tasks[todo_idx])
                print('now doing {}/{}'.format(todo_idx + 1, len(tasks)))
                thread = RunTasks_thread(todo_idx, tasks[todo_idx])
                task_doing_pool.append(thread)
                thread.start()
                todo_idx += 1
                if todo_idx >= len(tasks):
                    break
