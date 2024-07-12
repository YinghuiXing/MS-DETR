from os import path
from datasets.kaist import KaistDetection

import xml.etree.ElementTree as ET


class KAISTVOCDetection(KaistDetection):
    def __init__(self, roots, datasets_rgb, datasets_t, action, just_test=False):
        super(KAISTVOCDetection, self).__init__(roots, datasets_rgb, datasets_t, action, just_test=just_test)

    def load_annotation(self, anno_relative_path):
        annotation_path = path.join(self.root, anno_relative_path)

        xmlRoot = ET.parse(annotation_path).getroot()

        boxes = list()
        pede_types = list()
        pede_labels = list()
        occ_types = list()
        is_filter = list()

        for obj in xmlRoot.iter('object'):
            objClass = obj.find('name').text.lower().strip()
            pede_types.append(objClass)
            pede_labels.append(self.class2index[objClass])
            is_filter.append(False)
            occ_types.append(0)

            bbox = obj.find('bndbox')
            bndbox = list()

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)

            boxes.append(bndbox)


        results = dict()
        results['boxes'] = boxes
        results['pede_types'] = pede_types
        results['pede_labels'] = pede_labels
        results['is_filter'] = is_filter
        results['occ_types'] = occ_types

        return results


def generateLLVIPDatasets(root_dir, action='train'):
    import os
    files_path = os.path.join(root_dir, 'visible', action)
    files_names = os.listdir(files_path)

    rgb_list = list()
    t_list = list()
    for file_name in files_names:
        file_id = file_name.split('.')[0]
        anno_relative_path = os.path.join('Annotations', file_id + '.xml')
        file_relative_path_rgb = os.path.join('visible', action, file_name)
        file_relative_path_t = os.path.join('infrared', action, file_name)

        rgb_list.append(file_relative_path_rgb + ',' + anno_relative_path + '\n')
        t_list.append(file_relative_path_t + ',' + anno_relative_path + '\n')

    rgb_save_name = 'LLVIP_rgb_' + action + '.txt'
    t_save_name = 'LLVIP_t_' + action + '.txt'

    with open(rgb_save_name, 'w') as f:
        f.writelines(rgb_list)
    with open(t_save_name, 'w') as f:
        f.writelines(t_list)


if __name__ == '__main__':
    rgb_test = ['LLVIP_rgb_train', ]
    t_test = ['LLVIP_t_train', ]

    kaist_root = '/data/wangsong/datasets/LLVIP'
    train_det = KAISTVOCDetection(kaist_root, rgb_test, t_test, action='train', just_test=True)

    res = train_det[16]
    for k in res.keys():
        if k not in ['rgb_img', 't_img']:
            print(k, res[k])