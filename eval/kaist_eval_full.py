from eval.bbGt import *
import numpy as np
import matplotlib.pyplot as plt
import os
from evaluation_script.evaluation_script import draw_all
from evaluation_script.evaluation_script import evaluate as evaluate2
from matplotlib.backends.backend_pdf import PdfPages

# writened by nh,2021.7.26


def kaist_eval_full(detectionDir, gtDir, dataset_type='test', cvc14=False):
    savedPaths, isEmpty = aggregate_detections(detectionDir, dataset_type=dataset_type, cvc14=cvc14)

    if not cvc14:
        reason_all_results_path = savedPaths['test-all']
        all_path = list()
        # all_path.append('evaluation_script/state_of_arts/MLPD_result.txt')
        # all_path.append('evaluation_script/state_of_arts/ARCNN_result.txt')
        # all_path.append('evaluation_script/state_of_arts/CIAN_result.txt')
        # all_path.append('evaluation_script/state_of_arts/MSDS-RCNN_result.txt')
        # all_path.append('evaluation_script/state_of_arts/MBNet_result.txt')
        all_path.append(reason_all_results_path)

        phase = "Multispectral"
        results = [evaluate2('KAIST_annotation.json', rstFile, phase) for rstFile in all_path]
        results = sorted(results, key=lambda x: x['all'].summarize(0), reverse=True)
        results_img_path = os.path.join(os.path.dirname(reason_all_results_path), 'result.jpg')
        draw_all(results, filename=results_img_path)

    # test experiment settings
    # [test_setting_name, condition, height_range, occlusion_types]
    # condition: there are 'test_all', 'test_day' and 'test_night'
    # height_range: [1, 45], [45, 115] and [115, positive infinite] indicates far, medium and near objects respectively
    # occlusion_types: there are 'none', 'partial' and 'heavy'
    if not cvc14:
        testSettings = [
            ['Reasonable-all', 'test-all', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-day', 'test-day', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-night', 'test-night', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-near', 'test-all', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-medium', 'test-all', [55, 115], ['none', 'partial']],
            ['Reasonable-day-near', 'test-day', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-day-medium', 'test-day', [55, 115], ['none', 'partial']],
            ['Reasonable-night-near', 'test-night', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-night-medium', 'test-night', [55, 115], ['none', 'partial']],
            ['Scale=near', 'test-all', [115, float("inf")], ['none']],
            ['Scale=medium', 'test-all', [45, 115], ['none']],
            ['Scale=far', 'test-all', [1, 45], ['none']],
            ['Occ=none', 'test-all', [1, float("inf")], ['none']],
            ['Occ=partial', 'test-all', [1, float("inf")], ['partial']],
            ['Occ=heavy', 'test-all', [1, float("inf")], ['heavy']],
            ['all', 'test-all', [1,float("inf")], ['none', 'partial', 'heavy']],
            ['all-day', 'test-day', [1,float("inf")], ['none', 'partial', 'heavy']],
            ['all-night', 'test-night', [1,float("inf")], ['none', 'partial', 'heavy']],
        ]
    else:
        testSettings = [
            ['Reasonable-all', 'test-all', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-day', 'test-day', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-night', 'test-night', [55, float("inf")], ['none', 'partial']],
            ['Reasonable-near', 'test-all', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-medium', 'test-all', [55, 115], ['none', 'partial']],
            ['Reasonable-day-near', 'test-day', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-day-medium', 'test-day', [55, 115], ['none', 'partial']],
            ['Reasonable-night-near', 'test-night', [115, float("inf")], ['none', 'partial']],
            ['Reasonable-night-medium', 'test-night', [55, 115], ['none', 'partial']],
            ['Scale=near', 'test-all', [115, float("inf")], ['none']],
            ['Scale=medium', 'test-all', [45, 115], ['none']],
            ['Scale=far', 'test-all', [1, 45], ['none']],
            ['all', 'test-all', [1, float("inf")], ['none', 'partial']],
            ['all-day', 'test-day', [1, float("inf")], ['none', 'partial']],
            ['all-night', 'test-night', [1, float("inf")], ['none', 'partial']],
        ]


    result = dict()
    if (isEmpty[1]) or (isEmpty[2]):
        # no detection
        print('this epoch has 0 detection result')
        for setting in testSettings:
            result[setting[0]] = 1
    else:
        for setting in testSettings:
            res = evaluate(setting, gtDir, savedPaths, cvc14)
            result[setting[0]] = float(res['imp_mr'])

    return result


def draw_all(eval_results, filename='figure.jpg', setting=''):
    pdf = False
    if filename.endswith('.pdf'):
        pdf = True
    methods = list(eval_results.keys())
    for i in range(len(methods)):
        if methods[i] == 'MS-DETR':
            methods[i] = 'MS-DETR(ours)'

    # colors = [plt.cm.get_cmap('Paired')(ii)[:3] for ii in range(len(eval_results))]
    colors = [plt.cm.get_cmap('Paired')(ii)[:3] for ii in range(len(eval_results) + 1)]
    colors[10] = colors[11]
    del colors[11]

    eval_results_all = list(eval_results.values())

    if not pdf:
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))

        drawRoc(axes[0], eval_results_all, methods, colors)
        axes[0].set_title('All', fontsize=30, weight='bold')

        filename += '' if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.svg') or filename.endswith('.eps') else '.jpg'
        if filename.endswith('.svg'):
            plt.savefig(filename, dpi=600, format='svg')
        elif filename.endswith('.eps'):
            plt.savefig(filename, dpi=600, format='eps')
        else:
            plt.savefig(filename)
    else:
        with PdfPages(filename) as pdf:
            fig, axes = plt.subplots(1, 1, figsize=(15, 10))
            drawRoc(axes, eval_results_all, methods, colors)
            axes.set_title(setting, fontsize=30, weight='bold')

            pdf.savefig()
            plt.close()


def drawRoc(ax, eval_results, methods, colors):
    assert len(eval_results) == len(methods) == len(colors)

    for eval_result, method, color in zip(eval_results, methods, colors):
        mean_s = eval_result['imp_mr'] * 100
        xx = eval_result['roc'][1]
        yy = 1 - eval_result['roc'][2]

        ax.plot(xx, yy, color=color, linewidth=3, label=f'{mean_s:.2f}%, {method}')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()

    yt = [1, 5] + list(range(10, 60, 10)) + [64, 80]
    yticklabels = ['.{:02d}'.format(num) for num in yt]

    yt += [100]
    yt = [yy / 100.0 for yy in yt]
    yticklabels += [1]

    ax.set_yticks(yt)
    ax.set_yticklabels(yticklabels)
    ax.grid(which='major', axis='both')
    ax.set_ylim(0.01, 1)
    ax.set_xlim(2e-4, 50)
    ax.set_ylabel('Miss rate', fontsize=30, weight='bold')
    ax.set_xlabel('False positives per image', fontsize=30, weight='bold')


def aggregate_detections(dtDir, rebuild=True, dataset_type='test', cvc14=False):
    """
    aggregate the detection files according to certain condition such as test-all, test-day and test-night

    :param dtDir: The directory of detection results, for example:/home/wangsong/exp/rgbt_ped_dect/up-detr-results/exp2/det/checkpoint
    :param rebuild: If True, rebuild the aggregated detection files even these files already existed.
    :param dataset_type: If 'test', evaluate the public test dataset that contains 2252 images;
                            If 'val', evaluate the private val dataset that provided by ws and contains 834 images
    :return:
        savedPaths: {condition: aggregated detections path}
        isEmpty: [whether the num of detection bbox under certain condition is 0]
    """
    assert dataset_type in ['test', 'val']
    savedPaths = {}
    isEmpty = []
    conditions = ['test-all', 'test-day', 'test-night']
    for cond in conditions:
        fileName = os.path.split(dtDir)[-1] + '-' + cond + '.txt'
        fileDir = os.path.abspath(os.path.join(dtDir, '..'))
        filePath = os.path.join(fileDir, fileName)

        savedPaths[cond] = filePath

        if os.path.exists(filePath) and (not rebuild):
            continue

        # For KAIST test datasets
        # when setId is in [6, 7, 8], the images was captured during the day
        # when setId is in [9, 10, 11], the images was captured during the night
        # when setId is 6, the number of videos is 5 and so on
        # label one image every 20 frames
        if not cvc14:
            if dataset_type == 'test':
                if cond == 'test-all':
                    setIds = [6, 7, 8, 9, 10, 11]
                    skip = 20
                    videoNum = [5, 3, 3, 1, 2, 2]
                elif cond == 'test-day':
                    setIds = [6, 7, 8]
                    skip = 20
                    videoNum = [5, 3, 3]
                elif cond == 'test-night':
                    setIds = [9, 10, 11]
                    skip = 20
                    videoNum = [1, 2, 2]
            else:
                if cond == 'test-all':
                    setIds = [2, 3, 4]
                    videoNum = [2, 2, 2]
                elif cond == 'test-day':
                    setIds = [2]
                    videoNum = [2]
                elif cond == 'test-night':
                    setIds = [3, 4]
                    videoNum = [2, 2]
                skip = 2

            file = open(filePath, 'w+')
            detectionBBoxNum = 0

            num = 0
            for s in range(len(setIds)):
                for v in range(videoNum[s]):
                    for i in range(skip - 1, 99999, skip):
                        detectionFileName = 'set%02d_V%03d_I%05d.txt' % (setIds[s], v, i)  # e.g.set11_V001_I01279.txt
                        detectionFilePath = os.path.join(dtDir, detectionFileName)
                        if not os.path.exists(detectionFilePath):
                            continue
                        num += 1
                        x1, y1, x2, y2, score = [], [], [], [], []
                        detectionFile = open(detectionFilePath)

                        for detection in detectionFile:
                            detection = detection.strip().split(' ')

                            if len(detection) == 5:
                                x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection))
                            else:
                                x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection[1:]))

                            x1.append(x1_t)
                            x2.append(x2_t)
                            y1.append(y1_t)
                            y2.append(y2_t)
                            score.append(score_t)
                        for j in range(len(score)):
                            strInput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n' % (num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j])
                            detectionBBoxNum += 1
                            file.write(strInput)
        else:
            file = open(filePath, 'w+')
            detectionBBoxNum = 0
            num = 0
            for dir, sub_dirs, file_names in os.walk(dtDir):
                file_names.sort()
                for file_name in file_names:
                    day_or_night = file_name.split('_')[0]
                    if cond == 'test-day' and day_or_night == 'night':
                        continue
                    if cond == 'test-night' and day_or_night == 'day':
                        continue
                    detectionFilePath = os.path.join(dir, file_name)
                    if not os.path.exists(detectionFilePath):
                        raise RuntimeError
                    num += 1
                    x1, y1, x2, y2, score = [], [], [], [], []
                    detectionFile = open(detectionFilePath)
                    for detection in detectionFile:
                        detection = detection.strip().split(' ')

                        if len(detection) == 5:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection))
                        else:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection[1:]))
                        x1.append(x1_t)
                        x2.append(x2_t)
                        y1.append(y1_t)
                        y2.append(y2_t)
                        score.append(score_t)
                    for j in range(len(score)):
                        strInput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n' % (
                        num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j])
                        detectionBBoxNum += 1
                        file.write(strInput)

        if cond == 'test-all':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 2252, '{:d}'.format(num)
                else:
                    assert num == 834, '{:d}'.format(num)
        elif cond == 'test-day':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 1455, '{:d}'.format(num)
                else:
                    assert num == 534, '{:d}'.format(num)
        elif cond == 'test-night':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 797, '{:d}'.format(num)
                else:
                    assert num == 300, '{:d}'.format(num)

        file.close()
        if detectionBBoxNum == 0:
            isEmpty.append(True)
        else:
            isEmpty.append(False)

    return savedPaths, isEmpty


def evaluate(setting, gtDir, savedPaths, cvc14=False):
    """

    :param setting: test setting description
    :param gtDir: test dataset annotation files directory
    :param savedPaths: the path of aggregated detections for certain condition
    :return:
        result: test result for certain setting
    """
    result = dict()
    pows = np.arange(-2, 0.25, 0.25)  # [-2. -1.75 -1.5 -1.25 -1. -0.75 -0.5 -0.25 0.]
    ref = np.power(10, pows)  # [0.01 0.01778279 0.03162278 0.05623413 0.1 0.17782794 0.31622777 0.56234133 1.]

    settingName = setting[0]
    condition = setting[1]
    heightRange = setting[2]
    occlusionTypes = setting[3]

    savedPath = savedPaths[condition]

    gtLoadConstraints = {
        'labels': ['person', ],
        'otherLabels': ['people', 'person?', 'cyclist'],
        'hRng': heightRange,
        'xRng': [5, 635],
        'yRng': [5, 507] if not cvc14 else [5, 466],
        'vType': occlusionTypes
    }

    annotationDir = os.path.join(gtDir, condition, 'annotations')
    gtBBoxesList, dtBBoxesList = loadAll(annotationDir, savedPath, gtLoadConstraints, cvc14)

    gtsList = list()
    dtsList = list()
    for gtBBoxes, dtBBoxes in zip(gtBBoxesList, dtBBoxesList):
        gts, dts = evalRes(gtBBoxes, dtBBoxes)
        gtsList.append(gts)
        dtsList.append(dts)
    x_fppi, y_recall, scores, recall_ref = compRoc(gtsList, dtsList, 1, ref.copy())

    if x_fppi is None or y_recall is None or scores is None or recall_ref is None:
        strShow = '%-30s \t can\'t cal' % \
                  (setting[0])
        print(strShow)
        return result
    logMissRate_ori = np.exp(np.mean(np.log(np.maximum(1 - recall_ref, 1e-10))))
    roc_ori = [scores, x_fppi, y_recall]

    result['ori_miss'] = recall_ref
    result['ori_mr'] = logMissRate_ori
    result['roc'] = roc_ori

    # improved annotations
    improvedAnnotationDir = os.path.join(gtDir, condition, 'annotations_KAIST_test_set')
    gtBBoxesList, dtBBoxesList = loadAll(improvedAnnotationDir, savedPath, gtLoadConstraints, cvc14)
    gtsList = list()
    dtsList = list()
    for gtBBoxes, dtBBoxes in zip(gtBBoxesList, dtBBoxesList):
        gts, dts = evalRes(gtBBoxes, dtBBoxes)
        gtsList.append(gts)
        dtsList.append(dts)
    fp, tp, score, miss = compRoc(gtsList, dtsList, 1, ref.copy())
    miss_imp = np.exp(np.mean(np.log(np.maximum(1 - miss, 1e-10))))
    roc_imp = [score, fp, tp]

    result['imp_miss'] = miss
    result['imp_mr'] = miss_imp
    result['imp_roc'] = roc_imp

    strShow = '%-30s \t log-average miss rate = %02.2f%% (%02.2f%%) recall = %02.2f%% (%02.2f%%)' % \
              (settingName, logMissRate_ori * 100, miss_imp * 100, roc_ori[2][-1] * 100, roc_imp[2][-1] * 100)
    print(strShow)

    return result


if __name__ == '__main__':
    dir = '/home/wangsong/exp/rgbt_ped_dect/up-detr-results/exp2/det/checkpoint'
    aggregate_detections(dir)
