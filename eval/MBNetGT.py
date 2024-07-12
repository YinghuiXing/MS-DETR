import numpy as np
import os

kaist_numpy=np.load('../data/cache/MBNet_kaist_train_data.npy',encoding="latin1",allow_pickle=True)

train_data=kaist_numpy.tolist()
num_imgs_train = len(train_data)

outpath = '../data/KAIST/train_3/MBNet_annotations'

# for file in train_data:
#     file_path_split = file['filepath'].split('/')
#     file_name = file_path_split[3] + '_' + file_path_split[4] + '_' + file_path_split[6].replace('png', 'txt')
#
#     bboxes = file['bboxes']
#     bboxes[:, 2] -= bboxes[:, 0]
#     bboxes[:, 3] -= bboxes[:, 1]
#
#     person_np = np.repeat('person', len(bboxes), axis=0).reshape((-1, 1))
#
#     res_all = np.concatenate((person_np, bboxes), axis=-1).tolist()
#
#     res_path = os.path.join(outpath, file_name)
#     np.savetxt(res_path, np.array(res_all), fmt='%s')

    # print(file)

valid = 0
for file in train_data:
    if len(file['bboxes']) > 0:
        valid += 1

print(valid)