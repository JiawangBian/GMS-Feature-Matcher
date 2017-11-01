
import math
from enum import Enum

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

import time
import code


from gms_matcher import GmsMatcher
from gms_matcher import GmsRobe
from gms_matcher import DrawingType

def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))

def demo_match3():

    imgP = cv2.imread("../data/kf_1140.png") #curr
    imgC = cv2.imread("../data/kf_2583.png") #prev
    imgCm = cv2.imread("../data/kf_2581.png") #currm

    gms = GmsRobe()

    print 'Test gmsrobe.match3()'
    startT = time.time()
    pts_C, pts_P, pts_Cm = gms.match3( imgC, imgP, imgCm )
    print 'gmsrobe.match3 took (ms): %4.2f' %(1000.*(time.time() - startT ) )

    gridd = gms.plot_3way_match( imgC, pts_C,   imgP, pts_P,    imgCm, pts_Cm, show_random_points=40 )
    cv2.imshow( 'gridd', gridd )
    cv2.waitKey(0)



def demo_match2_guided():

    print 'Test gmsrobe.match2_guided()'
    imgP = cv2.imread("../data/kf_1140.png") #curr
    imgC = cv2.imread("../data/kf_2583.png") #prev

    orb = cv2.ORB_create(50)
    kp1 = orb.detect(imgC, None)
    print 'nORB Pts:', len(kp1)
    pts_C = np.transpose( np.array([ np.array(k.pt) for k in kp1 ]) ) #2xN


    gms = GmsRobe()

    startT = time.time()
    ptC, ptP = gms.match2_guided( imgC, pts_C, imgP )
    print 'Elapsed total (ms): %4.2f' %(1000.0 * (time.time() - startT )  )
    print 'nGuided Pts: ', ptC.shape[1]

    cv2.imshow( 'orb points on curr', gms.plot_points_on_image( imgC, pts_C ) )
    cv2.imshow( 'xcanvas', gms.plot_point_sets( imgC, ptC, imgP, ptP ) )
    cv2.waitKey(0)




def demo_match2():
    print 'Test simple gmsrobe.match2()'
    imgP = cv2.imread("../data/kf_1140.png") #curr
    imgC = cv2.imread("../data/kf_2583.png") #prev

    gms = GmsRobe()

    startT = time.time()
    ptC, ptP = gms.match2( imgC, imgP )
    # ptC, ptCm = gms.match2( imgC, imgCm )
    print 'Elapsed total (ms): %4.2f' %(1000.0 * (time.time() - startT )  )


    xcanvas = gms.plot_point_sets( imgC, ptC, imgP, ptP )

    # r = np.random.randint( 0, ptC.shape[1], 50 )
    # xcanvas = gms.plot_point_sets( imgC, ptC[:,r], imgCm, ptCm[:,r] )
    cv2.imshow( 'xcanvas', xcanvas )
    cv2.waitKey(0)



def demo_gms_original_implementation():
    print 'Test for Original GMS Implementation, ie. class GmsMatcher'
    # img1 = cv2.imread("../data/nn_left.jpg")
    # img2 = cv2.imread("../data/nn_right.jpg")

    # img1 = imresize(img1, 240)
    # img2 = imresize(img2, 240)

    img1 = cv2.imread("../data/kf_2581.png")
    img2 = cv2.imread("../data/kf_2583.png")



    # orb = cv2.ORB_create(10000)
    # orb.setFastThreshold(0)
    # if cv2.__version__.startswith('3'):
    #     matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # else:
    #     matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    # gms = GmsMatcher(orb, matcher)
    gms = GmsMatcher()
    startT = time.time()
    matches = gms.compute_matches(img1, img2)
    print 'gms.compute_matches took (ms): %4.2f' %(1000.*(time.time() - startT ) )
    # gms.draw_matches(img1, img2, DrawingType.ONLY_LINES)
    gms.draw_matches(img1, img2, DrawingType.POINTS_AND_TEXT)

if __name__ == '__main__':
    # demo_match3()
    # demo_match2_guided()
    demo_match2()
    # demo_gms_original_implementation()
