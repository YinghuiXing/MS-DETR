# -*- encoding: utf-8 -*-
'''
@File    :   select_no_people.py    
@Contact :   shaw@mail.nwpu.edu.cn
@License :   (C)Copyright 2019-2020, XiuWeiZhangGroup-CV-NWPU

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-12-12 下午5:11   ShawYun      1.0         None
'''

import os
from shutil import copyfile

# annotation_path = '/LabData/CSP_Fusion_V3/data/KAIST/test/annotations_improve_test'
# annotation_path = '/LabData/CSP_Fusion_V3/data/KAIST/train_3/annotations_new'
annotation_path = '/LabData/CSP_Fusion_V3/data/KAIST/train_3/annotation_2X'
# new_annotation_path = '/media/shawyun/000A53E4000C8A1A/Lab_Projects/KAIST_DataSet/annotations_improve_test_no_people'


def select_no_people_image(annotation_path,new_annotation_path ):
    file_no_people = []
    for home, dirs, files in os.walk(annotation_path):
        for file in files:
            file_path = os.path.join(annotation_path, file)
            with open(file_path,'r') as f:
                content = f.read()
                # if content.find('people') == -1:
                if content.find('people') != -1:
                    file_no_people.append(file_path)
                    # copyfile(src=file_path, dst=new_annotation_path +'/'+ file)
                    print(file)
                    # open(new_annotation_path +'/'+ file, 'w')
    print(len(file_no_people))


def remove_people_image(annotation_path, new_annotation_path):

    for home, dirs, files in os.walk(annotation_path):
        for file in files:
            file_path = os.path.join(annotation_path, file)
            new_file_path = os.path.join(new_annotation_path, file)
            # read the orginal file
            with open(file_path,'r') as f:
                orginal_content = f.readlines()
            # write the dest file
            string = str(orginal_content)
            if string.find('people') == -1:
                with open(new_file_path, 'w') as nf:
                    for line in orginal_content:
                        nf.write(line)


def remove_people_in_annotation(annotation_path, new_annotation_path):

    for home, dirs, files in os.walk(annotation_path):
        for file in files:
            file_path = os.path.join(annotation_path, file)
            new_file_path = os.path.join(new_annotation_path, file)
            # read the orginal file
            with open(file_path,'r') as f:

                orginal_content = f.readlines()
            # write the dest file
            with open(new_file_path, 'w') as nf:
                for line in orginal_content:
                    if line.find('people') == -1:
                        nf.write(line)
                    else:
                        print(file)


if __name__ == '__main__':
    # annotation_path = '/LabData/CSP_Fusion_V3/data/KAIST/train_3/annotations_new'
    # new_annotation_path = '/media/shawyun/000A53E4000C8A1A/Lab_Projects/KAIST_DataSet/shaw_kaist_annotations/annotations_improve_train_noPeople'
    # remove_people_in_image(annotation_path, new_annotation_path)
    # annotation_path = '/media/shawyun/000A53E4000C8A1A/Lab_Projects/KAIST_DataSet/annotations_new/sanitized_annotations/sanitized_annotations'
    # new_annotation_path = '/media/shawyun/000A53E4000C8A1A/Lab_Projects/KAIST_DataSet/shaw_kaist_annotations/annotations_improve_test_no_people_image'
    # remove_people_image(annotation_path, new_annotation_path)

    select_no_people_image('/media/shawyun/000A53E4000C8A1A/Lab_Projects/KAIST_DataSet/annotations_improve_test', '')
