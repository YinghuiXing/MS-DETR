import json
import matplotlib.pyplot as plt
import os

COLORS = ['red', 'blue', 'green', 'yellow', 'pink', 'black', 'orange', 'gray']


def draw_training_curve(path, total_epoch=100):
    log_file = open(path, 'r')

    epoch_list = list()
    for line in log_file:
        epoch_list.append(float(json.loads(line)['train_loss']))

    saved_name = 'TrainingCurve'

    plt.figure()
    x = range(0, total_epoch)
    plt.plot(x, epoch_list, color='red')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax = plt.gca()
    ax.set_title(saved_name)

    path = os.path.join(os.path.dirname(path), saved_name + '.jpg')
    plt.savefig(path)


def draw_training_curve_compare(path_list, epochs, descriptions):
    plt.figure()
    saved_name = 'TrainingCurveCompare'

    for i, (path, epoch, description) in enumerate(zip(path_list, epochs, descriptions)):
        log_file = open(path, 'r')

        epoch_list = list()
        for line in log_file:
            epoch_list.append(float(json.loads(line)['train_loss']))

        x = range(0, epoch)
        plt.plot(x, epoch_list, color=COLORS[i % len(COLORS)], label=description)

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax = plt.gca()
    ax.set_title('Training Cruve Compare')

    path = os.path.join(os.path.dirname(path), saved_name + '.jpg')
    plt.savefig(path)


def draw_training_val_curve(log_path, val_path, total_epoch=100):
    log_file = open(log_path, 'r')
    val_file = open(val_path, 'r')

    epoch_list = list()
    merit_all = list()
    merit_day = list()
    merit_night = list()
    # epoch_list_bottleneck = list()
    # epoch_list_t = list()
    for line1, line2 in zip(log_file, val_file):
        epoch_list.append(float(json.loads(line1)['train_loss']))
        merit_all.append(float(json.loads(line2)['Reasonable-all']) * 100)
        merit_day.append(float(json.loads(line2)['Reasonable-day']) * 100)
        merit_night.append(float(json.loads(line2)['Reasonable-night']) * 100)
        # epoch_list_bottleneck.append(float(json.loads(line)['train_loss_bottleneck']))
        # epoch_list_t.append(float(json.loads(line)['train_loss_t']))

    saved_name = 'TrainingValCurve'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = range(0, total_epoch)
    line1 = ax.plot(x, epoch_list, color='red', label='Loss')
    ax.set_title('Train&Val Curve')
    ax.set_xlabel("Epoch")
    ax.set_ylabel('Loss')
    # plt.plot(x, epoch_list_bottleneck, color='blue')
    # plt.plot(x, epoch_list_t, color='yellow')

    ax1 = ax.twinx()
    line2 = ax1.plot(x, merit_all, color='green', label='Resonable-all')
    line3 = ax1.plot(x, merit_day, color='blue', label='Resonable-day')
    line4 = ax1.plot(x, merit_night, color='yellow', label='Resonable-night')
    ax1.set_ylabel('average log miss rate')

    lines = line1 + line2 + line3 + line4

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels)

    path = os.path.join(os.path.dirname(log_path), saved_name + '.jpg')
    plt.savefig(path)


if __name__ == '__main__':
    t1 = '/data/wangsong/results/23_5_25/exp1/log.txt'
    t2 = '/data/wangsong/results/23_5_25/exp2/log.txt'
    t3 = '/data/wangsong/results/23_5_25/exp3/log.txt'
    t4 = '/data/wangsong/results/23_5_25/exp4/log.txt'
    t5 = '/data/wangsong/results/23_5_25/exp8/log.txt'
    t6 = '/data/wangsong/results/23_5_25/exp9/log.txt'
    t7 = '/data/wangsong/results/23_5_25/exp10/log.txt'
    t8 = '/data/wangsong/results/23_5_25/exp11/log.txt'
    t9 = '/data/wangsong/results/23_5_25/exp36/log.txt'
    t10 = '/data/wangsong/results/23_5_25/exp37/log.txt'
    t11 = '/data/wangsong/results/23_5_25/exp39/log.txt'
    t12 = '/data/wangsong/results/23_5_25/exp41/log.txt'
    t = [t9, t10, t11, t12]
    epochs = [20] * 3 + [10]
    descriptions = ['RGB', 'Thermal', 'finetune-rgb&thermal', 'finetune-rgb&thermal-0.1lr-10epoch']

    draw_training_curve_compare(t, epochs, descriptions)
    #draw_training_curve(t, total_epoch=60)

