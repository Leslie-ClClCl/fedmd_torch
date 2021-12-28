import matplotlib.pyplot as plt


def draw_loss(losses, description, saved_path=None):
    x = range(1, len(losses) + 1)
    y = losses
    fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小
    plt.plot(x, y, 'g-')
    plt.title(description)
    plt.xlabel('iters')
    plt.ylabel('loss')
    if saved_path is not None:
        plt.savefig(saved_path)
    plt.show()


if __name__ == '__main__':
    draw_loss([1, 0.5], 'test')
