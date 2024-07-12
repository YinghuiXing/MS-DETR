import os
from eval.kaist_eval_full import *
import csv

# writened by nh,2021.7.26

#detpath = 'your detection path'


def eval_all(gtdir, detpath):
    listd = os.listdir(detpath)
    listd.sort()
    print(listd)
    dirs = []
    for ld in listd:
        p = os.path.join(detpath, ld)
        if os.path.isdir(p):
            print(p)
            dirs.append(p)

    MR, recall,epoch = [], [],[]
    MR.append("MR")
    recall.append("Recall")
    epoch.append('')
    for i in range(0, len(dirs)):
        print('eval epoch ',i+1)
        res = kaist_eval_full(os.path.join(dirs[i], 'det'), gtdir, True, True)
        epoch.append('%d' % (i+1))
        if res is not None:
            MR.append(res[0]['imp_mr'])
            recall.append(res[0]['imp_roc'][2][-1])
        else:
            MR.append('emp')
            recall.append('emp')
    
    write = True
    if write:
        # 1. 创建文件对象
        f = open(os.path.join(detpath,'res.csv'), 'w', encoding='utf-8',newline='')
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        csv_writer.writerow(epoch)
        # 4. 写入csv文件内容
        csv_writer.writerow(MR)
        csv_writer.writerow(recall)


if __name__ == '__main__':
    print('test')
    gtDir = './KAISTdevkit-matlab-wrapper/improve_annotations_liu/'

    dtDir = 'E:/Workspace/Python/Object Detection/MBNet-master/data/result'
    mn = 'off2_fusion_p3p4_justconcate_illum_V3_7th_pixelAugment_bs15'
    detpath = './output/valresults/KAIST/h/%s_score_0.1_nms_0.5' % (mn)
    #kaist_eval_full(dtDir, gtDir, False, True)

    eval_all(gtDir, detpath)
