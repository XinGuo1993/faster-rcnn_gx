#!/usr/bin/python
# coding=utf-8
import json
import os
import cv2
from cv2 import cv
from math import sqrt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import cv2
import math
import sys
import random


d_angle = 15

def rotate_src(src, points, angle):
    center_x = (points[0][0] + points[1][0] + points[2][0] + points[3][0])/4
    center_y = (points[0][1] + points[1][1] + points[2][1] + points[3][1])/4
    center = (center_x, center_y)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(src, M, (src.shape[1],src.shape[0]))

    for i in range(4):
        x = points[i][0] - center[0]
        y = points[i][1] - center[1]
        points[i][0] = x * math.cos(math.radians(angle)) + y * math.sin(math.radians(angle)) + center_x
        points[i][1] = -x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)) + center_y
    lx=sys.maxint
    rx=0
    uy=sys.maxint
    dy=0
    for point in points:
        xtemp = point[0]
        ytemp = point[1]
        if xtemp <lx:
            lx = xtemp
        if xtemp > rx:
            rx = xtemp
        if ytemp < uy:
            uy = ytemp
        if ytemp > dy:
            dy = ytemp
    lx = max(lx, 0)
    rx = min(rx, dst.shape[1])
    uy = max(uy, 0)
    dy = min(dy, dst.shape[0])
    dst = dst[uy:dy, lx:rx]
    return dst
def percept(src, points, w2h):
    rate = 1
    rows,cols,ch = src.shape
    lx = cols
    rx = 0
    uy = rows
    dy = 0
    for point in points:
        if point[0] < lx:
            lx = point[0]
        if point[0] > rx:
            rx = point[0]
        if point[1] < uy:
            uy = point[1]
        if point[1] > dy:
            dy = point[1]
    width = rx - lx
    height = dy - uy
    temp_w2h = random.random() + w2h
    resize_ratio = (temp_w2h)/(width*1.0/height)
    src = cv2.resize(src,(int(cols*resize_ratio*rate),rows*rate))
    resized_points = []
    for point in points:
        resized_points.append([int(resize_ratio * point[0]*rate), point[1]*rate])
    
    rows,cols,ch = src.shape
    lx = cols
    rx = 0
    uy = rows
    dy = 0
    for point in resized_points:
        if point[0] < lx:
            lx = point[0]
        if point[0] > rx:
            rx = point[0]
        if point[1] < uy:
            uy = point[1]
        if point[1] > dy:
            dy = point[1]
    width = rx - lx
    height = dy - uy
    #
    center = [(lx + rx)/2, (uy + dy)/2]
    leftUpCorner = [max(center[0] - 2*width,0), max(center[1] - 2*width,0)]
    perceptionPoints = [0]*4
    pts1 = np.float32(resized_points)
    while True:
        direction = random.randint(0,1)
        #d_height = random.randint(0, int(0.2*height))
        #d_height = 0
        if direction == 0:
            perceptionPoints[0] = [rx-leftUpCorner[0], uy-leftUpCorner[1]]
            perceptionPoints[1] = [random.randint(int(rx-0.8 * height), rx)-leftUpCorner[0], dy-random.randint(0, int(0.3*height))-leftUpCorner[1]]
            perceptionPoints[2] = [random.randint(int(lx-0.8 * height), lx)-leftUpCorner[0], dy-random.randint(0, int(0.3*height))-leftUpCorner[1]]
            perceptionPoints[3] = [lx - leftUpCorner[0], uy-leftUpCorner[1]]
        else:
            perceptionPoints[0] = [rx-leftUpCorner[0], uy-leftUpCorner[1]]
            perceptionPoints[1] = [random.randint(rx, int(rx+0.8 * height))-leftUpCorner[0], dy-random.randint(0, int(0.3*height))-leftUpCorner[1]]
            perceptionPoints[2] = [random.randint(lx, int(lx+0.8 * height))-leftUpCorner[0], dy-random.randint(0, int(0.3*height))-leftUpCorner[1]]
            perceptionPoints[3] = [lx - leftUpCorner[0], uy-leftUpCorner[1]]

        if perceptionPoints[1][0] - perceptionPoints[2][0] > perceptionPoints[0][0] - perceptionPoints[3][0]:
            continue
        pts2 = np.float32(perceptionPoints)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        perception_size = (int(width*4), int(width*4))
        dst = cv2.warpPerspective(src, M, perception_size)
        break
    return [dst, perceptionPoints]


def align(src, points):
    rows,cols,ch = src.shape
    lx = cols
    rx = 0
    uy = rows
    dy = 0
    for point in points:
        if point[0] < lx:
            lx = point[0]
        if point[0] > rx:
            rx = point[0]
        if point[1] < uy:
            uy = point[1]
        if point[1] > dy:
            dy = point[1]
    perceptionPoints = [0]*4
    pts1 = np.float32(points)
    perceptionPoints[0] = [rx, uy]
    perceptionPoints[1] = [rx, dy]
    perceptionPoints[2] = [lx, dy]
    perceptionPoints[3] = [lx, uy]
    pts2 = np.float32(perceptionPoints)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src, M, (cols, rows))
    return [dst, perceptionPoints]


def rotate(src1, points1, angle):
    points = []
    for point in points1:
        points.append([point[0], point[1]])
    img = np.zeros((int(src1.shape[0]*2), int(src1.shape[1]*2), 3), np.uint8)
    corner = [(img.shape[0] - src1.shape[0])/2 , (img.shape[1] - src1.shape[1])/2]
    img[corner[0]:corner[0]+src1.shape[0],corner[1]:corner[1]+src1.shape[1]] = src1
    src = img
    for point in points:
        point[0] += corner[1]
        point[1] += corner[0]
    center_x = (points[0][0] + points[1][0] + points[2][0] + points[3][0])/4
    center_y = (points[0][1] + points[1][1] + points[2][1] + points[3][1])/4
    center = (center_x, center_y)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(src, M, (src.shape[1],src.shape[0]))

    for i in range(4):
        x = points[i][0] - center[0]
        y = points[i][1] - center[1]
        points[i][0] = x * math.cos(math.radians(angle)) + y * math.sin(math.radians(angle)) + center_x
        points[i][1] = -x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)) + center_y
    lx=sys.maxint
    rx=0
    uy=sys.maxint
    dy=0
    for point in points:
        xtemp = point[0]
        ytemp = point[1]
        if xtemp <lx:
            lx = xtemp
        if xtemp > rx:
            rx = xtemp
        if ytemp < uy:
            uy = ytemp
        if ytemp > dy:
            dy = ytemp

    lx = max(lx, 0)
    rx = min(rx, dst.shape[1])
    uy = max(uy, 0)
    dy = min(dy, dst.shape[0])
    width = rx - lx
    height = dy - uy
    border_x = random.randint(int(1*width/10), int(2*width/10))
    border_y = random.randint(int(1*height/10), int(2*height/10))
    lx = max(lx-border_x, 0)
    rx = min(rx+border_x, dst.shape[1])
    uy = max(uy-border_y, 0)
    dy = min(dy+border_y, dst.shape[0])
    dst = dst[uy:dy, lx:rx]
    return dst


def get_train_data(src, points, width, height, angle, number, w2h):
    result = []
    src, points = align(src, points)
    if angle == 0:
        img = rotate(src, points,0)
        img = cv2.resize(img, (width, height))
        result.append(img)
    num = 0
    while num < number:
        dst, points2 = percept(src, points, w2h)
        temp_angle = random.randint(-d_angle/2 + angle, d_angle/2 + angle)
        dst = rotate(dst, points2,temp_angle)
        dst = cv2.resize(dst, (width, height))
        result.append(dst)
        num += 1
    return result

def make_train_data(angle, width, height, w2h):
    f = open('data/train.dat', 'r')
    s1 = 'train_data/positives_octave_0.0'
    s2 = 'train_data/negatives_octave_0.0'
    s3 = 'train_data/hard_negatives_octave_0.0'
    if not os.path.exists(s1):
        os.makedirs(s1)
    if not os.path.exists(s2):
        os.makedirs(s2)
    if not os.path.exists(s3):
        os.makedirs(s3)
    num = 0
    for line in f:
        s=json.loads(line)
        # begin process s
        img_path = s['path']
        #print img_path
        img = cv2.imread(img_path)
        src = cv2.resize(img,(800,600))
        #
        src_n = src
        img_n = np.zeros((int(src_n.shape[0]*3), int(src_n.shape[1]*3), 3), np.uint8)
        corner_n = [(img_n.shape[0] - src_n.shape[0])/2 , (img_n.shape[1] - src_n.shape[1])/2]
        img_n[corner_n[0]:corner_n[0]+src_n.shape[0],corner_n[1]:corner_n[1]+src_n.shape[1]] = src_n
        #
        keypoints = s['keypoints']
        points = []
        for point in keypoints:
            xtemp = int(float(point.split(',')[0]))
            ytemp = int(float(point.split(',')[1]))
            points.append([xtemp,ytemp])
        number = 8
        plates = get_train_data(src,points,width,height,angle,number, w2h)
        plate_num = 0
        for plate in plates:
            resized_plate = plate
            img_name = 'train_data/positives_octave_0.0/positive_' + str(num) +'_'+str(plate_num)+ '.jpg'
            cv2.imwrite(img_name, resized_plate)
            ##
            """
            if num < 400:
                width = plate.shape[1]
                height = plate.shape[0]
                img_name = 'train_data/hard_negatives_octave_0.0/hard_negatives_1_' + str(num) + '_' + str(plate_num) + '.jpg'
                img_temp = cv2.resize(plate[0:height,width/2:width], (145, 60))
                cv2.imwrite(img_name, img_temp)
                img_name = 'train_data/hard_negatives_octave_0.0/hard_negatives_2_' + str(num) + '_' + str(plate_num) + '.jpg'
                img_temp = cv2.resize(plate[0:height,0:width/2], (145, 60))
                cv2.imwrite(img_name, img_temp)            
                img_name = 'train_data/hard_negatives_octave_0.0/hard_negatives_3_' + str(num) + '_' + str(plate_num) + '.jpg'
                img_temp = cv2.resize(plate[0:height,width/4:width], (145, 60))
                cv2.imwrite(img_name, img_temp)            
                img_name = 'train_data/hard_negatives_octave_0.0/hard_negatives_4_' + str(num) + '_' + str(plate_num) + '.jpg'
                img_temp = cv2.resize(plate[0:height,0:3*width/4], (145, 60))
                cv2.imwrite(img_name, img_temp)
            """
            ##
            plate_num += 1
        lx=sys.maxint
        rx=0
        uy=sys.maxint
        dy=0
        for point in keypoints:
            xtemp=int(float(point.split(',')[0]))
            ytemp=int(float(point.split(',')[1]))
            if xtemp <lx:
                lx = xtemp
            if xtemp > rx:
                rx = xtemp
            if ytemp < uy:
                uy = ytemp
            if ytemp > dy:
                dy = ytemp

        img_n[uy+corner_n[0]:dy+corner_n[0], lx+corner_n[1]:rx+corner_n[1]] = np.zeros((dy-uy, rx-lx, 3), img.dtype)
        points_n= [0]*4
        points_n[0] = [src_n.shape[1] + corner_n[1],corner_n[0]]
        points_n[1] = [src_n.shape[1] + corner_n[1], src_n.shape[0] + corner_n[0]]
        points_n[2] = [corner_n[1], src_n.shape[0] + corner_n[0]]
        points_n[3] = [corner_n[1],corner_n[0]]
        img_n = rotate_src(img_n, points_n, angle)      

        neg_name = 'train_data/negatives_octave_0.0/negative_' + str(num) + '.jpg'
        cv2.imwrite(neg_name, img_n)
        ##
        hard_points = [0]*4
        hard_number = 2
        if angle == 0:
            temp_num = 4
        else:
            temp_num = 2
        for hard_num in range(temp_num):
            if hard_num == 0:
                hard_points[0] = [(points[0][0] + points[3][0])/2, (points[0][1] + points[3][1])/2]
                hard_points[1] = [(points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2]
                hard_points[2] = [points[2][0], points[2][1]]
                hard_points[3] = [points[3][0], points[3][1]]
            elif hard_num == 1:
                hard_points[0] = [points[0][0], points[0][1]]
                hard_points[1] = [points[1][0], points[1][1]]
                hard_points[2] = [(points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2]
                hard_points[3] = [(points[0][0] + points[3][0])/2, (points[0][1] + points[3][1])/2]
            elif hard_num == 2:
                hard_points[0] = [points[0][0], points[0][1]]
                hard_points[1] = [(points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2]
                hard_points[2] = [(points[3][0] + points[2][0])/2, (points[3][1] + points[2][1])/2]
                hard_points[3] = [points[3][0], points[3][1]]
            elif hard_num == 3:
                hard_points[0] = [(points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2]
                hard_points[1] = [points[1][0], points[1][1]]
                hard_points[2] = [points[2][0], points[2][1]]
                hard_points[3] = [(points[3][0] + points[2][0])/2, (points[3][1] + points[2][1])/2]
            hard_plates = get_train_data(src,hard_points,width,height,angle,hard_number, w2h)
            num_hard_plate = 0
            for hard_plate in hard_plates:
                 img_name = 'train_data/hard_negatives_octave_0.0/hard_negatives_' + str(num) + '_' + str(num_hard_plate) +'_'+ str(hard_num) + '.jpg'
                 cv2.imwrite(img_name, hard_plate)
                 num_hard_plate += 1
        ##
        num += 1
        # end process s
def make_conf_file(name, angle, width, height):
    file1 = open('config.ini')
    file2 = open('temp.ini', 'w')
    if angle == 0:
        t_ratio = width*1.0/height
        min_ratio = max(round(2.0/t_ratio, 2), 0.3)
        max_ratio = round(7.0/t_ratio, 2)
        min_scale = round(max(30.0/height, 0.3/min_ratio), 2)
        max_scale = round(420.0/height, 2)
        for line in file1:
            if line == 'object_window=0,0,180,60\n':
                line = 'object_window=0,0,'+str(width)+','+str(height)+'\n'
            elif line == 'model_window=180,60\n':
                line = 'model_window='+str(width)+','+str(height)+'\n'
            elif line == 'output_model_filename=v0\n':
                line = 'output_model_filename=' + name +'\n'
            elif line == 'min_scale = 1\n':
                line = 'min_scale = ' + str(min_scale) + '\n'
            elif line == 'max_scale = 1\n':
                line = 'max_scale = ' + str(max_scale) + '\n'
            elif line == 'num_scales = 1\n':
                line = 'num_scales = 15\n'
            elif line == 'min_ratio = 1\n':
                line = 'min_ratio = ' + str(min_ratio) + '\n'
            elif line == 'max_ratio = 1\n':
                line = 'max_ratio = ' + str(max_ratio) + '\n'
            elif line == 'num_ratios = 1\n':
                line = 'num_ratios = 5\n' 
            file2.write(line)
    elif angle == 15 or angle == -15:
        t_ratio = width*1.0/height
        min_ratio = max(round(1.0/t_ratio,2), 0.3)
        max_ratio = round(4.0/t_ratio, 2)
        min_scale = round(max(30.0/height, 0.3/min_ratio),2)
        max_scale = round(420.0/height, 2)
        for line in file1:
            if line == 'object_window=0,0,180,60\n':
                line = 'object_window=0,0,'+str(width)+','+str(height)+'\n'
            elif line == 'model_window=180,60\n':
                line = 'model_window='+str(width)+','+str(height)+'\n'
            elif line == 'output_model_filename=v0\n':
                line = 'output_model_filename=' + name +'\n'
            elif line == 'min_scale = 1\n':
                line = 'min_scale = ' + str(min_scale) + '\n'
            elif line == 'max_scale = 1\n':
                line = 'max_scale = ' + str(max_scale) + '\n'
            elif line == 'num_scales = 1\n':
                line = 'num_scales = 15\n'
            elif line == 'min_ratio = 1\n':
                line = 'min_ratio = ' + str(min_ratio) + '\n'
            elif line == 'max_ratio = 1\n':
                line = 'max_ratio = ' + str(max_ratio) + '\n'
            elif line == 'num_ratios = 1\n':
                line = 'num_ratios = 5\n' 
            file2.write(line)
    elif angle == 30 or angle == -30:
        t_ratio = width*1.0/height
        min_ratio = round(1.0/t_ratio, 2)
        max_ratio = round(3.0/t_ratio, 2)
        min_scale = round(max(40.0/height, 0.3/min_ratio),2)
        max_scale = round(420.0/height, 2)
        for line in file1:
            if line == 'object_window=0,0,180,60\n':
                line = 'object_window=0,0,'+str(width)+','+str(height)+'\n'
            elif line == 'model_window=180,60\n':
                line = 'model_window='+str(width)+','+str(height)+'\n'
            elif line == 'output_model_filename=v0\n':
                line = 'output_model_filename=' + name +'\n'
            elif line == 'min_scale = 1\n':
                line = 'min_scale = ' + str(min_scale) + '\n'
            elif line == 'max_scale = 1\n':
                line = 'max_scale = ' + str(max_scale) + '\n'
            elif line == 'num_scales = 1\n':
                line = 'num_scales = 15\n'
            elif line == 'min_ratio = 1\n':
                line = 'min_ratio = ' + str(min_ratio) + '\n'
            elif line == 'max_ratio = 1\n':
                line = 'max_ratio = ' + str(max_ratio) + '\n'
            elif line == 'num_ratios = 1\n':
                line = 'num_ratios = 5\n' 
            file2.write(line)
    elif angle == 45 or angle == -45 or angle == 60 or angle == -60:
        t_ratio = width*1.0/height
        min_ratio = round(0.4/t_ratio,2)
        max_ratio = round(1.0/t_ratio, 2)
        min_scale = round(max(40.0/height, 0.3/min_ratio), 2)
        max_scale = round(500.0/height, 2)
        for line in file1:
            if line == 'object_window=0,0,180,60\n':
                line = 'object_window=0,0,'+str(width)+','+str(height)+'\n'
            elif line == 'model_window=180,60\n':
                line = 'model_window='+str(width)+','+str(height)+'\n'
            elif line == 'output_model_filename=v0\n':
                line = 'output_model_filename=' + name +'\n'
            elif line == 'min_scale = 1\n':
                line = 'min_scale = ' + str(min_scale) + '\n'
            elif line == 'max_scale = 1\n':
                line = 'max_scale = ' + str(max_scale) + '\n'
            elif line == 'num_scales = 1\n':
                line = 'num_scales = 15\n'
            elif line == 'min_ratio = 1\n':
                line = 'min_ratio = ' + str(min_ratio) + '\n'
            elif line == 'max_ratio = 1\n':
                line = 'max_ratio = ' + str(max_ratio) + '\n'
            elif line == 'num_ratios = 1\n':
                line = 'num_ratios = 5\n' 
            file2.write(line)
    file2.close()
    file1.close()
f = open('windows.txt', 'r')
for line in f:
    ss = line.split(' ')
    height = int(ss[3])
    make_train_data(int(ss[1]), int(ss[2]), int(ss[3]), int(ss[4]))
    make_conf_file(ss[0], int(ss[1]), int(ss[2]), int(ss[3]))
    os.system('./boosted_learning -c temp.ini')
    os.system('rm -r train_data')
    os.system('rm temp.ini')
