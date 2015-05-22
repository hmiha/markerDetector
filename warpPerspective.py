#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

### const ###
hue_threshold = 175
S_threshold   = 5000
rect_size     = 150

def get_reversed_hue_img(img):
    '''
    読み込んだBGR画像から，HSV画像に変換し，Hueだけの画像hueを作成します．
    赤はHueで0~10(deg.)付近で若干扱いにくいため，
    画像全体から最大値180(deg.)を引くことで，赤を170~180(deg.)付近に配置しなおしています．
    変換後の画像revをmainに返します．
    '''
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    rev = 180 - hue
    return rev

def get_hull(rev, img):
    '''
    まず，赤が170-180付近にマッピングされた画像revをしきい値処理し，
    しきい値以上の画素を180，しきい値以下の画素を0になるよう2値化し，それをthreshold_imgとします．
    次に，threshold_imgから輪郭抽出を行います．
    得られた輪郭controursを一つづつ確認し，面積が大きい物(ここでは5000としている)だった場合，
    その輪郭から凸包を計算します．
    その凸法をポリゴンで近似し，そのポリゴンが4点で表される場合，正方形と判断し，輪郭を描画します．
    返り値は，近似されたポリゴンの頂点座標です．
    '''
    retval, threshold_img = cv2.threshold(rev, hue_threshold, 180, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshold_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if  cv2.contourArea(cnt) > S_threshold:     # remove small areas to avoid noise
            hull = cv2.convexHull(cnt)              # find the convex hull of contour
            len_of_hull = 0.1 * cv2.arcLength(hull, True)
            approximated_hull = cv2.approxPolyDP(hull, len_of_hull, True)
            # the accuracy of approximated_hull is set to length of original contour*0.1

            if len(approximated_hull)==4:   # Is the approximated_hull is rectangle?
                cv2.drawContours(img, [approximated_hull], 0, (0,255,0), 2)
    cv2.imshow('win', img)
    cv2.waitKey(0)

    return approximated_hull

def get_corner(hull):
    '''
    頂点座標をfindHomography()で扱える形式に変換し，返します．
    '''
    corner_pt = np.float32(hull.reshape((4, 2)))
    return corner_pt

def get_trans_mat(corner_pt):
    '''
    Homography行列Hを計算し，返します．
    sizeは，変換後の正方形マーカの1辺のサイズです．
    Homography行列Hを返します．
    '''
    dst_pt    = np.float32([
                [rect_size,   0],
                [rect_size, rect_size],
                [0,   rect_size],
                [0,     0]
                ])

    H, Hstatus = cv2.findHomography(corner_pt, dst_pt, cv2.RANSAC)
    return H

def solve_result_img(H, img):
    '''
    計算されたHomography行列を用いて，射影変換を実行します．
    dispsizeは，変換後の画像の大きさです．
    '''
    dispsize = 1200
    dst = cv2.warpPerspective(img, H, (dispsize, dispsize))

    cv2.imshow('win', dst)
    cv2.waitKey(0) 

    return

if __name__ == '__main__':
    if(len(sys.argv) <> 2):
        print 'Usage: python warpPerspective.py <input_image_name>'
        print 'Exit.'
        sys.exit()

    _img = cv2.imread(sys.argv[1])
    img = _img.copy()

    rev  = get_reversed_hue_img(img)
    hull = get_hull(rev, img)
    corner_pt = get_corner(hull)

    H = get_trans_mat(corner_pt)
    solve_result_img(H, _img)
    print 'End.'
