import numpy as np
import cv2
import os
import io
import json
from matplotlib.pyplot import plot as plt
from PIL import Image
from shutil import copyfile
import pdb
from tqdm import tqdm
import glob
from random import shuffle
# image_path = 'frame1425.jpg'
# json_path = 'frame1425.json'


_W_ = 600
_H_ = 300
_X_min_ = 270
_X_max_ = 1240
_Y_ = 440
BLACK = [0,0,0]
_SIZE_1 = 400
_SIZE_2 = 400
def get_pascal_labels():
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


'''
parse json file with labels to image and matrix style

@input: 
image_path: image path
json_path : corresponding json path

@output:
labels: dictionary with instance labels. The key is the labels of instances such as '1' and '2' respectively corresponding to 1st instance  and 2nd instance. The value is a list in  the form of [semantic labels (numpy.array in the shape of image size. 0 means background, 1 means arm, 2 means hand, 3 means product) , key points' locations (numpy array in the shape of 6 x 2) ]
'''

def parse_json_label(image_path, json_path):
    with open(json_path, encoding='utf-8') as json_file:
        segment = json.loads(json_file.read())

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    labels = {}

    for objectList in segment['shapes']:
        pointsNP = np.asarray(objectList['points'], dtype=np.int32)
        name = objectList['label']
        name = name.lower()
        # print (pointsNP)
        inst = []
        part = []

        if 'arm' in name:
            part.append('arm' )
            inst.append ( name[3:])
        elif 'hand' in name:
            part.append ('hand')
            inst.append ( name[4:])
        elif 'product' in name:
            parts = name.split('/')
            for nameTmp in parts:
                part.append('product')
                inst.append( nameTmp[7:] )
        elif 'p' in name and '-' in name:
            tmp = name.split('-')
            part.append (  'kpt' + tmp[1] )
            inst.append ( name[1] )

        for j in range(len(inst)):
            instTmp = inst[j]
            partTmp = part[j]
            # print ()
            if instTmp not in labels:
                labels[instTmp] = {'arm':[],'hand':[], 'product':[], 'pt1':None, 'pt1':None, 'pt2':None, 'pt3':None, 'pt4':None, 'pt5':None, 'pt6':None,}
            if 'arm' == partTmp:
                labels[instTmp]['arm'].append(pointsNP)
            elif 'hand' == partTmp:
                labels[instTmp]['hand'].append(pointsNP)
            elif 'product' == partTmp:
                labels[instTmp]['product'].append (pointsNP)
            elif 'kpt' in partTmp:
                idx = partTmp[3:]
                # print (pointsNP)
                labels[instTmp]['pt' + idx] = (pointsNP[0][0],pointsNP[0][1])
        # name = name.split('_')
        # part = name[0]
        # inst = name[1]
        # if inst not in labels:
        #   labels[inst] = [ np.zeros([height,width], dtype = np.int32) , np.zeros([6,2] , dtype = np.int32) ]
        # if 'arm' == part:
        #   labels[inst][0] = cv2.fillPoly(labels[inst][0], [pointsNP], 1)
        # elif 'hand' == part:
        #   labels[inst][0] = cv2.fillPoly(labels[inst][0], [pointsNP], 2)
        # elif 'product' == part:
        #   labels[inst][0] = cv2.fillPoly(labels[inst][0], [pointsNP], 3)
        # elif 'kpt' == part:
        #   idx = int(name[2] )
        #   labels[inst][1][idx,:] = pointsNP
    return labels

def parse_json_label_v2(image_path, json_path):
    with io.open(json_path, encoding='utf-8', errors='ignore') as json_file:
        segment = json.loads(json_file.read())

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    labels = []
    find = False
    label_arm = False
    check_hand = False
    check_arm = False
    current_num = 4
    part = []
    for objectList in segment['shapes']:
        pointsNP = np.asarray(objectList['points'], dtype=np.int32)
        name = objectList['label']
        name = (name.lower()).replace(' ','').replace('phont','phone').replace('tabie','table')
        others = {'otherbag': [], 'bag': [], 'product': [], 'phone': [], 'receipt': []}
        num_bag = 0

        # if 'hand' in name and 'product' not in name:
        #     index = name[-1]
        #     if index not inat check_ph and index not in check_arm:
        #         return None, None, False

        # if it is on table

        hand = []

        if 'bag' in name:
            if 'mask' in name:
                continue
            else:
                print(name)
                part.append(pointsNP)
                num_bag += 1
                find = True
                continue
        elif 'hand' in name:
            if len(name) <=5:
                hand.append(pointsNP)



    return part, hand, find


def generate_mask(labels, hand, image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    mask = np.zeros([height, width])
    polys = labels
    mask = cv2.fillPoly(mask, [polys], 255)
    for poly_hand in hand:
        mask = cv2.fillPoly(mask, [poly_hand], 2)

    return mask


def decode_segmap(label_mask, dataset, plot=False):
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return (rgb * 255 ).astype(np.uint8)

def generate_mask_all(out_dir):
    #subidxs = ['20190326_132','20190326_119','20190326_448/001','20190326_448/003','20190326_448/004','20190326_448/005','20190218214416640','20190219103711391']
    #subidxs = ['0404_1/save20190313','0404_1/save20190313-2','0404_1/save20190313-5','0404_1/save20190313-6','0404_1/save20190313-7','0404_1/save20190313-9','0404_2/save20190313-3','0404_2/save20190313-4','0404_3']
    #subidxs = ['20190328/abnormal_1_decode','20190328/abnormal_2_decode','20190328/abnormal_3_decode','20190328/abnormal_noReflection_1_decode','20190328/abnormal_noReflection_2_decode','20190328/abnormal_noReflection_3_decode','20190328/normalScan_decode','20190409/save1_decode']
    subidxs = ['data_0404/imageset']
    #subidxs = ['case5_decode', 'case6_decode', 'case7_decode', 'case8_decode', 'case9_decode', 'case10_decode', 'case11_decode', 'case12_decode']
    new_path_jpg = '/Users/shiyaoliao/Documents/checkout-data/label_0401/RGB_depth/bag/JPEG/'
    new_path_cat = '/Users/shiyaoliao/Documents/checkout-data/label_0401/RGB_depth/bag/Segmentationclass/'

    if not os.path.exists(new_path_jpg):
        os.makedirs(new_path_jpg)

    if not os.path.exists(new_path_cat):
        os.makedirs(new_path_cat)
    count = 0

    for subidx in subidxs:
        data_path = '/Users/shiyaoliao/Documents/checkout-data/label_0401/labeled_0401/' + subidx + '/'

        frames = [x[:-4] for x in os.listdir(data_path) if '.jpg' in x]
        for frame in tqdm(frames):
            image_path = data_path + frame + '.jpg'
            json_path = data_path + frame + '.json'

            if os.path.isfile(image_path) and os.path.isfile(json_path):

                label_frame, hand, isContinue = parse_json_label_v2(image_path, json_path)
                if isContinue:
                    index = 0
                    for pointsNP in label_frame:
                        index += 1

                        label_mask = generate_mask(pointsNP, hand, image_path)
                        label_rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
                        label_rgb[:, :, 0] = label_mask.copy()
                        label_rgb[:, :, 1] = label_mask.copy()
                        label_rgb[:, :, 2] = label_mask.copy()
                        label_mask = label_rgb.copy()
                        # label_mask = decode_segmap(label_mask, 'pascal')

                        img = cv2.imread(image_path)
                        #img = img[0:_Y_,_X_min_:_X_max_,:]
                        #label_mask = label_mask[0:_Y_,_X_min_:_X_max_,:]
                        h,w,_ = img.shape
                        pad = int((w - h)/2)

                        img = cv2.copyMakeBorder(img,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)
                        label_mask = cv2.copyMakeBorder(label_mask,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)

                        img = cv2.resize(img,(_SIZE_2,_SIZE_1), interpolation=cv2.INTER_NEAREST)
                        label_mask = cv2.resize(label_mask,(_SIZE_2,_SIZE_1), interpolation=cv2.INTER_NEAREST)
                        img = img.astype(np.uint8)
                        label_mask = label_mask.astype(np.uint8)
                        # cv2.imwrite(new_path_jpg + frame + '.jpg', img)
                        # cv2.imwrite(new_path_cat + frame + '.png', label_mask[...,::-1])
                        cv2.imwrite(new_path_jpg + subidx.replace('/', '_') + '_' + frame + '_' + str(index) + '.jpg', img)
                        cv2.imwrite(new_path_cat + subidx.replace('/', '_') + '_' + frame + '_' + str(index) + '.png',label_mask)

                        #Add by Shiyao, check data label
                        # data_label = cv2.addWeighted(img, 0.5, label_mask, 0.5, 0)
                        # cv2.imwrite('/Users/shiyaoliao/Documents/checkout-data/label_0401/RGB_depth/check-label-838/' + subidx.replace('/','_') + '_' + frame + '.jpg',data_label)
                        count += 1

    print("Total we process %d images." % count)


def generate_image_sets(data_dir,out_dir):
    label_list = glob.glob(data_dir+'/*.png')
    for i in range(len(label_list)):
        label_list[i] = label_list[i].replace(data_dir,"").replace(".png","").replace("/","").replace(".jpg","")
    shuffle(label_list)
    train_num = int(len(label_list)*0.85)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_ = open(os.path.join(out_dir,'train.txt'),'w')
    test_ = open(os.path.join(out_dir,'val.txt'),'w')
    for i in range(len(label_list)):
        if i < train_num:
            train_.write(label_list[i]+'\n')
        else:
            test_.write(label_list[i]+'\n')
    train_.close()
    test_.close()


out_dir = '/Users/shiyaoliao/Documents/checkout-data/label_0401/RGB_depth/SegmentationClass_armandhand/'
set_dir = '/Users/shiyaoliao/Documents/checkout-data/label_0401/RGB_depth/Segmentation_armandhand'
generate_mask_all(out_dir)
#generate_image_sets(out_dir,set_dir)