import itertools
import os
import subprocess
import time

import pytest

models = ['resnet18']
N_alignment = 40000
private_classes = ['18', '30', '60']
public_classes = 10
N_rounds = [1]
N_logits_matching_round = [300]
N_private_training_round = [4]
with_reverse = ['0']
repeat = [i for i in range(0, 2)]
result_save_dir = ["../result"]


@pytest.mark.parametrize('repeat,with_reverse,private_classes',
                         itertools.product(repeat, with_reverse, private_classes))
def test_p(repeat, with_reverse, private_classes):
    time.sleep(int((repeat + private_classes) / 5) + 5)
    result_saved_path = '../results/' + str(models[0]) + '_' + str(private_classes) + '/'
    if not os.path.exists(result_saved_path):
        os.mkdir(result_saved_path)
    result_saved_path += str(repeat) + '/'
    if not os.path.exists(result_saved_path):
        os.mkdir(result_saved_path)
    model_saved_names = []
    for model in models:
        model_saved_names.append(model)
    info = model_saved_names[0] + '-' + str(with_reverse)
    filename = result_saved_path + '/' + info + '.log'
    log_file = open(filename, 'w')
    subprocess.call(["python",
                     "./CIFAR_Balanced.py",
                     "-models", *map(lambda x: str(x), models),
                     "-private_classes", private_classes,
                     "-public_classes", public_classes,
                     "-result_saved_path", result_saved_path,
                     "-train_private_model", '1',
                     "-with_reverse", with_reverse
                     ], stdout=log_file, stderr=log_file)
    log_file.close()


if __name__ == '__main__':
    pytest.main(['-sv', '--tests-per-worker=1'])
