import os
import xml.etree.ElementTree as ET

CLASS_TXT = []
CLASS_XML = []

def read_txt(file_path):
    with open(file_path) as f:
        content = f.readlines()
    result = []
    if len(content) > 1:
        for ind in range(1, len(content)):
            single_content = content[ind].split()
            if single_content[0] not in CLASS_TXT:
                CLASS_TXT.append(single_content[0])
            result.append([single_content[0], int(single_content[1]), int(single_content[2]), int(single_content[3]), int(single_content[4])])

    result.sort(key=lambda x:x[1])
    return result


def read_xml(file_path):
    result = []
    xmlRoot = ET.parse(file_path).getroot()

    for obj in xmlRoot.iter('object'):
        temp = []
        objClass = obj.find('name').text.lower().strip()
        if objClass not in CLASS_XML:
            CLASS_XML.append(objClass)

        temp.append(objClass)
        bbox = obj.find('bndbox')
        temp.append(int(float(bbox.find('xmin').text)))
        temp.append(int(float(bbox.find('ymin').text)))
        temp.append(int(float(bbox.find('xmax').text)) - int(float(bbox.find('xmin').text)))
        temp.append(int(float(bbox.find('ymax').text)) - int(float(bbox.find('ymin').text)))

        result.append(temp)

    result.sort(key=lambda x: x[1])
    return result


def compare_list(a, b):
    if len(a) != len(b):
        print('两个文件的目标数量不一致')
        return False
    for a_temp, b_temp in zip(a, b):
        if a_temp[1] != b_temp[1]:
            print('xmin不一致, %d != %d' %(a_temp[1], b_temp[1]))
            return False
        if a_temp[2] != b_temp[2]:
            print('ymin不一致, %d != %d' %(a_temp[2], b_temp[2]))
            return False
        if a_temp[4] != b_temp[4]:
            print('h不一致, %d != %d' %(a_temp[4], b_temp[4]))
            return False
        if a_temp[3] != b_temp[3]:
            print('w不一致, %d != %d xmin=%d' %(a_temp[3], b_temp[3], a_temp[1]))
            return False
    return True


if __name__ == '__main__':
    dir1 = '/data/wangsong/datasets/KAIST/gt/sanitized_annotations'  # txt
    dir2 = '/data/wangsong/datasets/KAIST/gt/sanitized_annotations_xml_all'  # xml

    count = 0
    count_txt = 0
    count_xml = 0
    for home, dirs, files in os.walk(dir1):
        for file in files:
            file_path_1 = os.path.join(dir1, file)
            file_path_2 = os.path.join(dir2, file.replace('txt', 'xml'))
            result1 = read_txt(file_path_1)
            result2 = read_xml(file_path_2)
            count_txt += len(result1)
            count_xml += len(result2)
            if not compare_list(result1, result2):
                print(file)
                count += 1
    print(count)
    print(CLASS_XML)
    print(CLASS_TXT)
    print(count_txt)
    print(count_xml)