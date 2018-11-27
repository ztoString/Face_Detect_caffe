# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#import Image
import sys
import os
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
import random
caffe_root = '/home/zt/caffe/'

sys.path.insert(0, caffe_root + 'python')
#os.environ['GLOG_minloglevel'] = '2'
import caffe
#caffe.set_device(0)
#caffe.set_mode_gpu()



class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rect(object):
    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        '''
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top    = max(p1.y, p2.y)

    def __str__(self):
        return "Rect[%d, %d, %d, %d]" % ( self.left, self.top, self.right, self.bottom )

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def rect_overlaps(r1,r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)

def rect_merge(r1,r2, mergeThresh):
    if rect_overlaps(r1,r2):
        SI= abs(min(r1.right, r2.right) - max(r1.left, r2.left)) * abs(max(r1.bottom, r2.bottom) - min(r1.top, r2.top))
        SA = abs(r1.right - r1.left)*abs(r1.bottom - r1.top)
        SB = abs(r2.right - r2.left)*abs(r2.bottom - r2.top)
        S=SA+SB-SI
        ratio = float(SI) / float(S)
        if ratio > mergeThresh :
            return 1
    return 0






# 获得一些人脸检测的边界框
def generateBoundingBox(featureMap, scale):
    boundingBox = []
    # 观察AlexNet网络，计算总步长stride=4*2*2*2=32
    # stride在此表示的是伸缩变换的总倍数，在还原图像的过程中使用
    stride = 32
    # 经过伸缩变换后的图像大小
    cellSize = 227
    #227 x 227 cell, stride=32
    # 从特征图中获得图像左上角的坐标以及识别为人脸的概率值
    for (x,y), prob in np.ndenumerate(featureMap):
        if(prob >= 0.95):
            print prob
	    # 将当前坐标（左上角，右下角）变换为原始图像坐标
            boundingBox.append([float(stride * y)/ scale, float(x * stride)/scale, float(stride * y + cellSize - 1)/scale, float(stride * x + cellSize - 1)/scale, prob])
    return boundingBox

def nms_average(boxes, groupThresh=2, overlapThresh=0.2):
    rects = []
    temp_boxes = []
    weightslist = []
    new_rects = []
    for i in range(len(boxes)):
        if boxes[i][4] > 0.2:
            rects.append([boxes[i,0], boxes[i,1], boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1]])

    rects, weights = cv2.groupRectangles(rects, groupThresh, overlapThresh)
    #######################test#########
    rectangles = []
    for i in range(len(rects)):
        # A______
        # |      |
        # -------B

    #                       A                                       B
        testRect = Rect( Point(rects[i,0], rects[i,1]), Point(rects[i,0]+rects[i,2], rects[i,1]+rects[i,3]))
        rectangles.append(testRect)
    clusters = []
    for rect in rectangles:
        matched = 0
        for cluster in clusters:
            if (rect_merge( rect, cluster , 0.2) ):
                matched=1
                cluster.left   =  (cluster.left + rect.left   )/2
                cluster.right  = ( cluster.right+  rect.right  )/2
                cluster.top    = ( cluster.top+    rect.top    )/2
                cluster.bottom = ( cluster.bottom+ rect.bottom )/2

        if ( not matched ):
            clusters.append( rect )
    result_boxes = []
    for i in range(len(clusters)):
        result_boxes.append([clusters[i].left, clusters[i].bottom, clusters[i].right, clusters[i].top, 1])
    return result_boxes


def face_detection(imgFile):
    net_full_conv = caffe.Net('/home/zt/face_detect/deploy_full_conv.prototxt',
                              '/home/zt/face_detect/model/alexnet_iter_50000_full_conv.caffemodel',
                              caffe.TEST)
    randNum = random.randint(1,10000)
    
    
    # 多尺寸变换
    scales = []
    factor = 0.793700526
    
    img = cv2.imread(imgFile)
    print img.shape
    
    # 指定图像最大尺寸
    largest = min(2, 4000/max(img.shape[0:2]))
    scale = largest
    # 指定图像最小尺寸
    minD = largest*min(img.shape[0:2])
    # 图像尺寸大于227时，将图像放入，进行伸缩变换
    while minD >= 227:
        scales.append(scale)
        scale *= factor
        minD *= factor
	
    # 存储人脸框
    total_boxes = []

    # 输入数据预处理，使用caffe.io.Transformer来实现
    for scale in scales:
        # 对于每一个scale值，都需要resize到相应大小
        scale_img = cv2.resize(img,((int(img.shape[0] * scale), int(img.shape[1] * scale))))
	# 将scale_img图像进行保存
        cv2.imwrite('/home/zt/face_detect/scale_img.jpg',scale_img)
		
        im = caffe.io.load_image('/home/zt/face_detect/scale_img.jpg')


	# reshape相应的data数据层
        net_full_conv.blobs['data'].reshape(1,3,scale_img.shape[1],scale_img.shape[0])
	# 构建transformer来处理data数据层
        transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))	# 对每一个channel通道进行减mean值操作
        transformer.set_transpose('data', (2,0,1))	# 将图像通道移动到最外层（h*w*c --> c*h*w）
        transformer.set_channel_swap('data', (2,1,0))	# 将输入图像 RGB 格式变为 BGR
        transformer.set_raw_scale('data', 255.0)	# 将像素点从 [0, 1] 变为 [0, 255]

	# 进行前向传播并输出每个滑动窗口属于人脸的概率值
        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
        print out['prob'][0,1].shape
		
	# 输入特征图的概率矩阵和scale，得到一些框
        boxes = generateBoundingBox(out['prob'][0,1], scale)
		
	# 如果人脸框不为空
        if(boxes):
            total_boxes.extend(boxes)

    # NMS非极大值抑制
    print total_boxes
    boxes_nms = np.array(total_boxes)
    true_boxes = nms_average(boxes_nms, 1, 0.2)
    if not true_boxes == []:
        (x1, y1, x2, y2) = true_boxes[0][:-1]
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0))
        win = cv2.namedWindow('test win', flags=0)  
  	cv2.imwrite("/home/zt/face_detect/result.jpg", img)
        cv2.imshow('test win', img)  
          
        cv2.waitKey(0)  
	
	

if __name__ == "__main__":
    imgFile = '/home/zt/face_detect/tmp9055.jpg'
    face_detection(imgFile)
