import os, glob
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify

import pdb

IMAGE_SIZE = (640, 512)     # KAIST Multispectral Benchmark

'''
txt annos:
% Each object struct has the following fields:
%  lbl  - a string label describing object type (eg: 'pedestrian')
%  bb   - [l t w h]: bb indicating predicted object extent
%  occ  - 0/1 value indicating if bb is occluded
%  bbv  - [l t w h]: bb indicating visible region (may be [0 0 0 0])
%  ign  - 0/1 value indicating bb was marked as ignore
%  ang  - [0-360] orientation of bb in degrees
'''

def txt_anno2dict(txt_file, sub_dir):
    vid_name = os.path.splitext(os.path.basename(txt_file))[0]
    # object info in each frame: id, pos, occlusion, lock, posv

    frame_name = vid_name
    anno = defaultdict(list)
    anno["id"] = frame_name
    with open(txt_file, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            datas = line.split(' ')
            classlbl = datas[0]
            bbox = datas[1:5]
            occl = datas[6]
            anno["label"].append(classlbl)
            anno["occlusion"].append(occl)
            anno["bbox"].append(bbox)

    return anno

def instance2xml_base(anno, img_size, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('KAIST Multispectral Ped Benchmark'),
        E.filename(anno['id']),
        E.source(
            E.database('KAIST pedestrian'),
            E.annotation('KAIST pedestrian'),
            E.image('KAIST pedestrian'),
            E.url('https://soonminhwang.github.io/rgbt-ped-detection/')
        ),
        E.size(
            E.width(img_size[0]),
            E.height(img_size[1]),
            E.depth(4)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin+w
            ymax = ymin+h
        else:
            xmin, ymin, xmax, ymax = bbox

        E = objectify.ElementMaker(annotate=False)        

        anno_tree.append(
            E.object(
            E.name(anno['label'][index]),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmax),
                E.ymax(ymax)
            ),
            E.pose('unknown'),
            E.truncated(0),
            E.difficult(0),
            E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree

def parse_anno_file(vbb_inputdir, xml_outputdir):

    xml_outputdir_l = xml_outputdir+'lwir'
    xml_outputdir_v = xml_outputdir+'vis'
    if not os.path.exists(xml_outputdir_l):
        os.makedirs(xml_outputdir_l)
    if not os.path.exists(xml_outputdir_v):
        os.makedirs(xml_outputdir_v)
    # annotation sub-directories in hda annotation input directory
    vbb_inputdir_l = vbb_inputdir+'lwir'
    vbb_inputdir_v = vbb_inputdir+'vis'
    assert os.path.exists(vbb_inputdir_l)
    assert os.path.exists(vbb_inputdir_v)
    sub_dirs_l = os.listdir(vbb_inputdir_l)
    sub_dirs_v = os.listdir(vbb_inputdir_v)
    file_obj = open('./train_both.txt','w+')
    a = 0
    # 反正原始标注中可见光和红外标注文件数量都是相同的，就只遍历一个获取文件名就OK了。
    for file in sub_dirs_l:
        #print( "Parsing annotations (txt): {}".format(filename) )
        vbb_file_l = os.path.join(vbb_inputdir_l, file)
        vbb_file_v = os.path.join(vbb_inputdir_v, file)

        anno_l = txt_anno2dict(vbb_file_l, file)
        anno_v = txt_anno2dict(vbb_file_v, file)
        if (anno_l is not None) and (anno_v is not None):
            A = anno_l.get('bbox')
            B = anno_v.get('bbox')
            if (A is None) or (B is None):
                continue
            file_obj.write(os.path.splitext(file)[0] + "\n")
            a+=1
            anno_tree_l = instance2xml_base(anno_l, IMAGE_SIZE)
            anno_tree_v = instance2xml_base(anno_v, IMAGE_SIZE)
            outfile_v = os.path.join(xml_outputdir_v, os.path.splitext(file)[0] + ".xml")
            outfile_l = os.path.join(xml_outputdir_l, os.path.splitext(file)[0] + ".xml")

            #print("outfile_v: {}".format(outfile_v))
            etree.ElementTree(anno_tree_l).write(outfile_l, pretty_print=True)
            etree.ElementTree(anno_tree_v).write(outfile_v, pretty_print=True)
    file_obj.close()
    print('{} files contain objs'.format(a))


if __name__ == "__main__":
    db_path = '.'#os.path.join(os.path.dirname(__file__), '..', 'kaist-rgbt')
    
    vbb_inputdir = os.path.join(db_path, 'anno_form/')
    xml_outputdir = os.path.join(db_path, 'anno_form-xml/')

    #os.system( 'curl -O http://multispectral.kaist.ac.kr/pedestrian/data-kaist/annotations-vbb.zip')
    #os.system( 'unzip -d %s/annotations-vbb -q annotations-vbb.zip' % db_path )
    parse_anno_file(vbb_inputdir, xml_outputdir) 