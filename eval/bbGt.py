import os
import numpy as np


# writened by nh,2021.7.26

class OBJ(object):
    def __init__(self, label, bbox, occlusion, bbv, ignore, angle):
        self.label = label
        self.bbox = bbox
        self.occlusion = occlusion
        self.bbv = bbv
        self.ignore = ignore if ignore is not None else 0
        self.angle = angle if angle is not None else 0


def loadAll(gtDir, dtPath, gtLoadConstraints, cvc14=False):
    """
    load all gt bounding boxes and detection bounding boxes
    :param gtDir: annotation directory
    :param dtPath: aggregated detections file path
    :param gtLoadConstraints: the constraints when loads gt bounding boxes
    :return:
        gtBBoxes: the list of gt bounding boxes in certain annotation file
        dtBBoxes: the list of det bounding boxes corresponding to some annotation file
    """
    gtBBoxes = list()
    dtBBoxes = list()

    assert gtDir is not None, 'The gtDir should not be None!'
    gtFilePaths, gtFileNames = getFiles(gtDir)

    for path in gtFilePaths:
        gtBBoxes.append(load_gt_bbox(path, gtLoadConstraints, cvc14)[1])

    if os.path.exists(dtPath):
        detData = np.loadtxt(dtPath, delimiter=',')

        for num in range(len(gtFileNames)):
            idx = detData[:, 0] == (num + 1)
            dtBBoxes.append(detData[idx][:, 1:])
    else:
        print(dtPath)
        raise RuntimeError('dtPath does not exist')
        dtBBoxes = None

    return gtBBoxes, dtBBoxes


def evalRes(gtBBoxes, dtBBoxes, threshold=0.5, multipleMatch=False):
    """ Evaluates detections against ground truth data.(from matlab)
     Uses modified Pascal criteria that allows for "ignore" regions. The
     Pascal criteria states that a ground truth bounding box (gtBb) and a
     detected bounding box (dtBb) match if their overlap area (oa):
      oa(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(union(gtBb,dtBb))
     is over a sufficient threshold (typically .5). In the modified criteria,
     the dtBb can match any subregion of a gtBb set to "ignore". Choosing
     gtBb' in gtBb that most closely matches dtBb can be done by using
     gtBb'=intersect(dtBb,gtBb). Computing oa(gtBb',dtBb) is equivalent to
      oa'(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(dtBb)
     For gtBb set to ignore the above formula for oa is used.

     Highest scoring detections are matched first. Matches to standard,
     (non-ignore) gtBb are preferred. Each dtBb and gtBb may be matched at
     most once, except for ignore-gtBb which can be matched multiple times.
     Unmatched dtBb are false-positives, unmatched gtBb are false-negatives.
     Each match between a dtBb and gtBb is a true-positive, except matches
     between dtBb and ignore-gtBb which do not affect the evaluation criteria.

     In addition to taking gt/dt results on a single image, evalRes() can take
     cell arrays of gt/dt bbs, in which case evaluation proceeds on each
     element. Use bbGt>loadAll() to load gt/dt for multiple images.

     Each gt/dt output row has a flag match that is either -1/0/1:
      for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
      for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]

     INPUTS
      gtBBoxes  - [mx5] ground truth array with rows [x y w h ignore]
      dtBBoxes  - [nx5] detection results array with rows [x y w h score]
      threshold  - [.5] the threshold on oa for comparing two bbs
      multipleMatch  - [0] if true allow multiple matches to each gt

     OUTPUTS
      gts   - [mx5] ground truth results [x y w h match]
      dts   - [nx6] detection results [x y w h score match]
    """
    if gtBBoxes is None:
        gtBBoxes = np.zeros((0, 5))
    if dtBBoxes is None:
        dtBBoxes = np.zeros((0, 5))

    gt_num = gtBBoxes.shape[0]
    dt_num = dtBBoxes.shape[0]

    # sort dt highest score first
    idx = np.argsort(-dtBBoxes[:, -1])
    dts = dtBBoxes[idx, :].copy()

    # sort gt ignore last, gtBBoxes按照ignore进行排序，即非ignore的排在前面，ignore的排在后面
    idx = np.argsort(gtBBoxes[:, -1])
    gts = gtBBoxes[idx, :].copy()

    # initialize gt match field, ignore(1) to ignore(-1) and not ignore(0) to unmatched(0)
    gts[:, 4] = -gts[:, 4]

    # dt add one column to indicate match status, initialize to unmatched(0)
    zs = np.zeros(dt_num)
    dts = np.column_stack((dts, zs))

    # Attempt to match each (sorted) dt to each (sorted) gt
    # oa size:col:num_gt row:num_dt
    oa = computeIOU(dts[:, :4], gts[:, :4], gts[:, 4] == -1)

    for i in range(dt_num):
        bestIOU = threshold
        bestGT = 0  # info about which gt bounding box is the best matcher for current dt bounding box
        matchFlag = 0  # This flag indicates the current bounding box whether matches any gt bounding box
        for j in range(gt_num):
            gtMatchFlag = gts[j, 4]
            if gtMatchFlag == 1 and ~multipleMatch:  # if this gt already matched, continue to next gt
                continue
            if matchFlag == 1 and gtMatchFlag == -1:  # if dt already matched, and on ignore gt, nothing more to do 如果当前dt已经匹配上某一个gt，并且当前gt已经是ignore了（后面的gt则全部是ignore），循环结束
                break
            # compute overlap area,continue to next gt unless better match made
            if oa[i, j] < bestIOU:
                continue
            # match successful and best so far, store appropriately
            bestIOU = oa[i, j]
            bestGT = j
            if gtMatchFlag == 0:
                matchFlag = 1
            else:
                matchFlag = -1  # 如果dt和gt的IOU > bestIOU, 但当前的gt已经匹配，则dt会被暂时认定为ignore，如果后面循环再也找不到一个非ignore gt与之匹配，则认定为ignore

        # store type of match for both dt and gt
        dts[i, 5] = matchFlag

        if matchFlag == 1:
            gts[bestGT, 4] = 1

    # valid_list = list()
    # for i in range(len(dts)):
    #     h = dts[i][3]
    #     score = dts[i][4]
    #     match = dts[i][5]
    #     if not match:
    #         pass
    #     else:
    #         valid_list.append(i)
    # dts = dts[valid_list]
    #
    #
    # for dt in dts:
    #     h = dt[3]
    #     score = dt[4]
    #     match = dt[5]
    #
    #     if h < 115 and match and score < 0.8:
    #         dt[4] = 0.8
    return gts, dts


def compRoc(gtsList, dtsList, roc=1, ref=[]):
    """
    % Compute ROC or PR based on outputs of evalRes on multiple images.
    %
    % ROC="Receiver operating characteristic"; PR="Precision Recall"
    % Also computes result at reference points (ref):
    %  which for ROC curves is the *detection* rate at reference *FPPI*
    %  which for PR curves is the *precision* at reference *recall*
    % Note, FPPI="false positive per image"
    %
    % USAGE
    %  xs,ys,score,ref = compRoc( gt, dt, roc, ref )
    %
    % INPUTS
    %  gtsList    - nx first output of evalRes()
    %  dtsList    - nx second output of evalRes()
    %  roc        - [1] if 1 compue ROC else compute PR
    %  ref        - [] reference points for ROC or PR curve
    %
    % OUTPUTS
    %  xs         - x coords for curve: ROC->FPPI; PR->recall
    %  ys         - y coords for curve: ROC->TP; PR->precision
    %  score      - detection scores corresponding to each (x,y)
    %  ref        - recall or precision at each reference point
    %
    % EXAMPLE
    """
    nImg = len(gtsList)
    gts = np.concatenate(gtsList)
    gts = gts[gts[:, 4] != -1, :]

    dts = np.concatenate(dtsList)
    dts = dts[dts[:, 5] != -1, :]

    if dts.shape[0] == 0:
        return None, None, None, None

    m = len(ref)
    noIgnoreNum = gts.shape[0]
    scores = dts[:, 4]
    matches = dts[:, 5]

    order = np.argsort(-scores)
    scores = scores[order]
    matches = matches[order]

    falsePositive = (matches != 1).astype(float)
    falsePositive = np.cumsum(falsePositive)
    truePositive = np.cumsum(matches)

    if roc:
        xs = falsePositive / nImg
        ys = truePositive / noIgnoreNum

        xs1 = np.concatenate([[-float('inf')], xs])
        ys1 = np.concatenate([[0], ys])

        for i in range(m):
            j = np.argwhere(xs1 <= ref[i])
            ref[i] = ys1[j[-1]]
    else:
        xs = truePositive / noIgnoreNum
        ys = truePositive / (falsePositive + truePositive)

        xs1 = np.concatenate([xs, [float('inf')]])
        ys1 = np.concatenate([ys, [0]])
        for i in range(m):
            j = np.argwhere(xs1 >= ref[i])
            ref[i] = ys1[j[1]]

    return xs, ys, scores, ref


def get_annotation_version(description):
    """
    get annotation version
    :param description: the first line of annotation file, e.g.'% bbGt version=3'
    :return:
        annotation_version: may be 0, 1, 2 or 3
    """
    if 'bbGt' in description:
        return int(description[-1])
    else:
        return 0


def get_field_number(version):
    """
    get number of fields for certain version
    :param version: annotation version, may be 0, 1, 2 or 3
    :return:
        number of fields for certain version
    """
    assert version in [0, 1, 2, 3], 'The annotation version is illegal!'
    ms = [10, 10, 11, 12]

    return ms[version]


def load_gt_bbox(gtFilePath, gtLoadConstraints, cvc14=False):
    """
    load gt bbox of certain gt file. The most important function is capture the ignore property for each box
    :param gtFilePath: annotation file path
    :param gtLoadConstraints: the constraints when loads gt bounding boxes
    :return:
        objs: list of bbox object in current gt file
        gtBBoxes: np.array or None, size is N * 5
    """
    objs, gtBBoxes = [], []

    labels = gtLoadConstraints['labels']
    otherLabels = gtLoadConstraints['otherLabels']
    hRng = gtLoadConstraints['hRng']
    wRng = None
    xRng = gtLoadConstraints['xRng']
    yRng = gtLoadConstraints['yRng']
    vType = gtLoadConstraints['vType']

    # using number to indicate occlusion types in test setting
    # 1 for 'None', 2 for 'partial' and 4 for 'heavy'
    vVal = 0
    if 'none' in vType:
        vVal += 1
    if 'partial' in vType:
        vVal += 2
    if 'heavy' in vType:
        vVal += 4

    gtFile = open(gtFilePath)
    annotationVersion = get_annotation_version(gtFile.readline().strip())
    fieldNum = get_field_number(annotationVersion)
    for line in gtFile:
        annotation = line.strip().split()
        bbox = [float(a) for a in annotation[1:5]]
        if cvc14:
            bbox[1] = bbox[1] / 1.087044832
            bbox[3] = bbox[3] / 1.087044832

        bbv = [float(a) for a in annotation[6:10]]
        label = annotation[0]
        occlusion = int(annotation[5])

        if fieldNum >= 11:
            ignore = int(annotation[10])
        if fieldNum >= 12:
            angle = annotation[11] if len(annotation) == 12 else None

        obj = OBJ(label, bbox, occlusion, bbv, ignore, angle)

        if obj.label not in labels and obj.label not in otherLabels:
            continue

        if otherLabels is not None:
            obj.ignore = obj.ignore or (obj.label in otherLabels)
        if xRng is not None:
            obj.ignore = obj.ignore or (obj.bbox[0] < xRng[0]) or (obj.bbox[0] > xRng[1])
            x2 = obj.bbox[0] + obj.bbox[2]
            obj.ignore = obj.ignore or (x2 < xRng[0]) or (x2 > xRng[1])
        if yRng is not None:
            obj.ignore = obj.ignore or (obj.bbox[1] < yRng[0]) or (obj.bbox[1] > yRng[1])
            y2 = obj.bbox[1] + obj.bbox[3]
            obj.ignore = obj.ignore or (y2 < yRng[0]) or (y2 > yRng[1])
        if wRng is not None:
            obj.ignore = obj.ignore or (obj.bbox[2] < wRng[0]) or (obj.bbox[2] > wRng[1])
        if hRng is not None:
            obj.ignore = obj.ignore or (obj.bbox[3] < hRng[0]) or (obj.bbox[3] > hRng[1])
        if vType is not None:
            obj.occlusion = pow(2, obj.occlusion)
            ak = int(bin(obj.occlusion & vVal), 2)
            obj.ignore = obj.ignore or (ak == 0)

        bbox.append(float(obj.ignore))
        objs.append(obj)
        gtBBoxes.append(bbox)

    gtFile.close()
    if len(objs) == 0:
        gtBBoxes = None
    else:
        gtBBoxes = np.array(gtBBoxes)
    return objs, gtBBoxes


def getFiles(directory):
    """
    get the files' names and paths below directory mentioned by parameter
    :param directory:
    :return:
        fileNames: the list of files' names below directory
        filePaths: the list of files' paths below directory
    """

    fileNames = os.listdir(directory)
    fileNames.sort()
    filePaths = [os.path.join(directory, name) for name in fileNames]

    return filePaths, fileNames


def getIOU(dt, gt):
    """
    calculate the iou between gt and dt bounding box
    :param dt: [x, y, w, h], (x, y) is the coordinate of the point on the upper left of the bounding box
    :param gt: [x, y, w, h], (x, y) is the coordinate of the point on the upper left of the bounding box
    :return:
        iou: intersection over Union
    """
    x1_d = dt[0]
    y1_d = dt[1]
    x2_d = dt[0] + dt[2]
    y2_d = dt[1] + dt[3]

    x1_g = gt[0]
    y1_g = gt[1]
    x2_g = gt[0] + gt[2]
    y2_g = gt[1] + gt[3]

    xA = max(x1_d, x1_g)
    yA = max(y1_d, y1_g)

    xB = min(x2_d, x2_g)
    yB = min(y2_d, y2_g)

    # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    area_d = dt[2] * dt[3]
    area_g = gt[2] * gt[3]

    iou = interArea / (area_d + area_g - interArea)

    return iou, interArea


def computeIOU(dts, gts, ignore=None):
    """
    get IOU matrix of dts and gts
    In the modified criteria, a gt bb may be marked as "ignore", in which
    case the dt bb can can match any subregion of the gt bb. Choosing gt' in
    gt that most closely matches dt can be done using gt'=intersect(dt,gt).
    Computing IOU(gt',dt) is equivalent to:
    IOU'(gt,dt) = area(intersect(gt,dt)) / area(dt)
    :param dts: [mx4], detection bounding boxes
    :param gts: [nx4], gt bounding boxes
    :param ignore: [n], ignore status for certain gt bounding boxes
    :return:
        iou: [mxn], the iou of detection bounding boxes and gt bounding boxes
    """
    d_num = dts.shape[0]
    g_num = gts.shape[0]
    iou = np.zeros((d_num, g_num))

    if ignore is None:
        ignore = np.zeros((g_num, 1))

    for i in range(d_num):
        for j in range(g_num):
            if ignore[j]:
                iou[i][j] = getIOU(dts[i], gts[j])[1] / (dts[i][2] * dts[i][3])
            else:
                iou[i][j] = getIOU(dts[i], gts[j])[0]
    return iou


if __name__ == '__main__':
    a = '/home/wangsong/datasets/KAIST/gt/improve_annotations_liu/test-all/annotations/set08_V001_I02699.txt'
    b = {
        'labels': ['person', ],
        'otherLabels': ['people', 'person?', 'cyclist'],
        'hRng': [55, float("inf")],
        'xRng': [5, 635],
        'yRng': [5, 507],
        'vType': ['none', 'partial']
    }

    c, d = load_gt_bbox(a, b)

    print(c)
    print(d)