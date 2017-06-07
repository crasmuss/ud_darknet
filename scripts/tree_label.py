import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=['train', 'test']

classes = ["tree"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('data/trees/Annotations/%s.xml'%(image_id))
    out_file = open('data/trees/labels/%s.txt'%(image_id), 'w')
    # size, width, height not in my annotations files -- oops!
    tree=ET.parse(in_file)
    root = tree.getroot()
    w = 1280
    h = 960

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for image_set in sets:
    if not os.path.exists('data/trees/Annotations/labels'):
        os.makedirs('data/trees/Annotations/labels')
    image_ids = open('data/trees/ImageSets/%s.txt'%(image_set)).read().strip().split()
    list_file = open('tree_%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/data/trees/JPEGImages/%s.jpg\n'%(wd, image_id))
        convert_annotation(image_id)
    list_file.close()

