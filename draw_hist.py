import matplotlib.pyplot as plt
import matplotlib
import numpy as np


fine_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


def draw_hist(data_lists, label_list=None, info=None, filename=None):
    x = range(len(label_list))
    x = [i*2.5 for i in x]
    plt.figure(num=1, figsize=(len(label_list)*2, 10))
    for idx in range(len(data_lists)):
        rects = plt.bar(x=[i+0.6*idx for i in x], height=data_lists[idx], width=0.6, alpha=0.8, label='net'+str(idx))
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{height:.2f}", ha="center", va="bottom")
    plt.ylim(0, 1)
    plt.ylabel('rate')
    plt.xticks([index + 0.4 for index in x], label_list)
    plt.title(info)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def draw_hist_old(num_list1, num_list2, label_list=None, info=None, filename=None):
    if label_list is None:
        label_list = [i for i in range(0, len(num_list1))]    # 横坐标刻度显示值
    x = range(len(num_list1))
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    plt.figure(num=1, figsize=(len(label_list), 10))
    rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="net1")
    rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="net2")
    plt.ylim(0, 1)     # y轴取值范围
    plt.ylabel("rate")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    if info is not None:
        plt.title(info)
    else:
        plt.title("correct rate")
    plt.legend()     # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, str(height), ha="center", va="bottom")
    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    num_list1 = [0.79, 0.72, 0.9, 0.95, 0.89, 0.96, 0.91, 0.86, 0.96, 0.91, 0.95, 0.94, 0.94, 0.83, 0.92]

    with_reverse = [
        [0.79, 0.72, 0.91, 0.92, 0.86, 0.96, 0.93, 0.75, 0.96, 0.72, 0.79, 0.88, 0.89, 0.77, 0.85]
    ]
    with_reverse = np.array(with_reverse)
    without_rev = [
        [0.79, 0.73, 0.9, 0.94, 0.85, 0.95, 0.94, 0.77, 0.96, 0.72, 0.79, 0.9, 0.92, 0.79, 0.87]
    ]
    without_rev = np.array(without_rev)
    with_reverse_mean = np.mean(with_reverse, axis=0)
    without_rev_mean = np.mean(without_rev, axis=0)
    fine_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                        'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                        'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm']
    label_pri = [3, 13, 19, 58, 81, 8, 14, 77, 88,92, 0, 5, 12, 36, 23]
    label_list = []
    for i in range(len(label_pri)):
        idx = label_pri[i]
        label_list.append(str(idx) + ' ' + fine_label_names[label_pri[i]])
    draw_hist([num_list1, without_rev_mean, with_reverse_mean], label_list)
