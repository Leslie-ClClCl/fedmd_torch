import subprocess
import time

import pynvml

pynvml.nvmlInit()
DEVICE_NUM = 2


def getGPUmem(idx):
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / (1024 * 1024)


def doTask(task, stdout, stderr):
    subprocess.call(task, stdout=stdout, stderr=stderr)


if __name__ == '__main__':
    tasks = [
        ['-models', 'resnet50', '-private_classes', '30', '-result_saved_path', 'results'],
        ['-models', 'resnet50', '-private_classes', '60', '-result_saved_path', 'results'],
        ['-models', 'resnet50', '-private_classes', '18', '-result_saved_path', 'results'],
        ['-models', 'resnet50', '-private_classes', '6', '-result_saved_path', 'results'],
    ]
    todo_idx = 0
    while todo_idx < len(tasks):
        for idx in range(DEVICE_NUM):
            freeMem = getGPUmem(idx)
            if freeMem > 10000:
                print(tasks[todo_idx])
                print('now doing {}/{}'.format(todo_idx + 1, len(tasks)))
                doTask(tasks[todo_idx])
                todo_idx += 1
                time.sleep(30)
            time.sleep(30)
