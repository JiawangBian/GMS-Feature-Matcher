## Adopted from : https://github.com/JiawangBian/GMS-Feature-Matcher
## Acknowledment to original Authors
# @inproceedings{bian2017gms,
#   title={GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence},
#   author={JiaWang Bian and Wen-Yan Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
#   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#   year={2017}
# }
#
#
# Added class GmsRobe by mpkuse
#   This class can be used to expand macthes, and to find 3way matches. the
#   core matcher is based on GMS-Feature-Matcher by Jiawang Bian.
#       Date: 1st Nov, 2017



import math
from enum import Enum

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

import time
import code

THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]


class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    POINTS_AND_TEXT = 3


class GmsMatcher:
    def __init__(self, descriptor=None, matcher=None):

        if descriptor is None:
            descriptor = cv2.ORB_create(100)
            descriptor.setFastThreshold(0)

        if matcher is None:
            if cv2.__version__.startswith('3'):
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            else:
                matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []
        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0
        # Grid Size
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

        self.descriptor = descriptor
        self.matcher = matcher
        self.gms_matches = []
        self.keypoints_image1 = []
        self.keypoints_image2 = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

    def compute_matches(self, img1, img2):
        startKeypts = time.time()
        self.keypoints_image1, descriptors_image1 = self.descriptor.detectAndCompute(img1, None )#np.array([]))
        self.keypoints_image2, descriptors_image2 = self.descriptor.detectAndCompute(img2, None )#np.array([]))
        print 'compute_matches(): detectAndCompute took (ms): %4.2f' %(1000.*(time.time() - startKeypts ) )

        size1 = Size(img1.shape[1], img1.shape[0])
        size2 = Size(img2.shape[1], img2.shape[0])

        if self.gms_matches:
            self.empty_matches()

        startMatcher = time.time()
        all_matches = self.matcher.match(descriptors_image1, descriptors_image2)
        code.interact( local=locals() )
        print 'compute_matches(): self.matcher.match took (ms): %4.2f' %(1000.*(time.time() - startMatcher ) )

        self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
        self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)

        self.matches_number = len(all_matches)
        self.convert_matches(all_matches, self.matches)
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)

        startVote = time.time()
        mask, num_inliers = self.get_inlier_mask(False, False) #This is the most expensive function call, which inturn calls run()
        print('Found', num_inliers, 'matches')
        print 'compute_matches(): GMS Voting took (ms): %4.2f' %(1000.*(time.time() - startVote ) )


        for i in range(len(mask)):
            if mask[i]:
                self.gms_matches.append(all_matches[i])
        return self.gms_matches

    # Normalize Key points to range (0-1)
    def normalize_points(self, kp, size, npts):
        for keypoint in kp:
            npts.append((keypoint.pt[0] / size.width, keypoint.pt[1] / size.height))

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0

        if not with_scale and not with_rotation:
            self.set_scale(0)
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_scale and not with_rotation:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2

    def draw_matches(self, src1, src2, drawing_type):
        height = max(src1.shape[0], src2.shape[0])
        width = src1.shape[1] + src2.shape[1]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:src1.shape[0], 0:src1.shape[1]] = src1
        output[0:src2.shape[0], src1.shape[1]:] = src2[:]

        if drawing_type == DrawingType.ONLY_LINES:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

        elif drawing_type == DrawingType.LINES_AND_POINTS:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
                cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

        elif drawing_type == DrawingType.POINTS_AND_TEXT:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255))
                cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0) )
                cv2.putText(output, str(i), tuple(map(int, left)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )
                cv2.putText(output, str(i), tuple(map(int, right)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )

        cv2.imshow('show', output)
        cv2.waitKey()
        return output


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height



class GmsRobe:
    ## This class is suppose to extent the functonality of GmsMatcher. All of the
    ## Following functions return 2xN matrix with (x,y) of matches. No image resizing
    ##      a) match2( imC, imP )
    ##      b) match3( imC, imP, imCm )
    ##      c) match2_guided( imC, pt1, imP )
    def __init__(self):
        self.gms = GmsMatcher()


    def match2( self, imC, imP ):
        """ Given current image(imC) and prev image(imP) returns GMS matches co-ordinates (x,y)_i
            Essentially this is a thin wrapper around the original GmsMatcher class.

            Typical usage with 240x320 images takes ~ 350 ms (Brute Force matcher)
        """
        matches = self.gms.compute_matches(imC, imP)
        return self._matches_to_cords( matches, self.gms.keypoints_image1, self.gms.keypoints_image2 )

    def match2_guided( self, imC, pts_C, imP ):
        """ Given current image (imC), with pts_C 2xN as input points on imC, the
        objective is to find these points in previous image (imP)

        The way we do this is to first compute all the matches between imC and imP.
        Then filter these matches to include only those pysically close to input points.

        Typical usage with 240x320 images and pts_C ~ 50 takes 354 ms overall.
        Part-A: Takes 352ms (Brute Force matcher)
        Part-B: Takes 2ms
        """

        timeA = time.time()
        gmsC, gmsP = self.match2( imC, imP )
        # cv2.imshow( 'org GMS C--P', self.plot_point_sets( imC, gmsC, imP, gmsP ) )

        print 'ElapsedA (ms): %4.2f' %(1000.0 * (time.time() - timeA ))


        # Now go thru each of pts_C, find nn of each of pts_C in gmsC. Only retain
        # the pair gmsC_i <--> gmsP_i if gmsC_i is within 1px of pts_C_i.
        L = []
        timeB = time.time()
        for i in range( pts_C.shape[1] ):

            # find nearest neighbour of pts_C_i in gmsC. TODO Consider using FLANN for this.
            diff = gmsC - np.expand_dims( pts_C[:,i],1 ) #2xN
            diff_norm = np.linalg.norm( diff, axis=0 ) #1xN
            minval = diff_norm.min()
            minarg = diff_norm.argmin() #This is essentially like 1-NN


            if minval < 1.0 :
                # print minval, minarg
                L.append( minarg )

        print 'ElapsedB (ms): %4.2f' %(1000.0 * (time.time() - timeB ))

        # L is list of index on gmsC <--> gmsP which are found in pts_C.
        # code.interact(local=locals() )
        return gmsC[:,L], gmsP[:,L]



    def match3( self, imC, imP, imCm ):
        """ To find 3way correspondences. Note: The order in which the images is given is critical
                imC  : current image
                imP  : previous image
                imCm : current-1 image


                Takes about 775 ms.
                Part-A: 727 ms (Brute Force matcher)
                Part-B: 47 ms
        """

        # Get C<-->P and C<-->Cm
        startA = time.time()
        _gmsC, _gmsP = self.match2( imC, imP ) #2xN, 2xN
        __gmsC, __gmsCm = self.match2( imC, imCm ) # This will be a larger set than _gmsC <--> _gmsP. 2xN, 2xN
        print 'gmsrobe.match3.A took (ms): %4.2f' %(1000.*(time.time() - startA ) )


        # print 'C<-- %d -->P' %(_gmsC.shape[1])
        # print 'C<-- %d -->Cm' %(__gmsC.shape[1])

        # r = np.random.randint( 0, __gmsC.shape[1], 50 )
        # cv2.imshow( 'C--P',  gms.plot_point_sets( imC, _gmsC, imP, _gmsP ) )
        # cv2.imshow( 'C--Cm', self.plot_point_sets( imC, __gmsC, imCm, __gmsCm ) )
        # # cv2.imshow( 'C', self.plot_points_on_image(imC, __gmsC[:,r]) )
        # # cv2.imshow( 'Cm', self.plot_points_on_image(imCm, __gmsCm[:,r]) )
        # cv2.waitKey(0)



        # loop thru _gmsC <--> _gmsCm. Find nearest cords of _gmsC_i in __gmsC.
        startB = time.time()
        L1 = []
        L2 = []
        for i in range( _gmsC.shape[1] ):
            diff = __gmsC - np.expand_dims(  _gmsC[:,i]  ,1 ) #2xN
            diff_norm = np.linalg.norm( diff, axis=0 ) #1xN
            minval = diff_norm.min()
            minarg = diff_norm.argmin() #This is essentially like 1-NN

            if minval < 1.0:
                # print i, minarg
                L1.append(i)
                L2.append(minarg)

        print 'gmsrobe.match3.B took (ms): %4.2f' %(1000.*(time.time() - startB ) )
        return _gmsC[:,L1], _gmsP[:,L1], __gmsCm[:,L2]






    ##################################################
    ################# Utilities  #####################
    ##################################################
    def _matches_to_cords( self, matches, kp1, kp2 ):
        """ Given the DMatches array and keypoints, returns list of cords"""
        N = len(matches)
        cords_1 = np.zeros( (2,N)  )
        cords_2 = np.zeros( (2,N)  )
        for i in range(N):
            left  = kp1[matches[i].queryIdx].pt
            right = kp2[matches[i].trainIdx].pt

            cords_1[0,i] = left[0]
            cords_1[1,i] = left[1]

            cords_2[0,i] = right[0]
            cords_2[1,i] = right[1]

            # print '---',i
            # print left
            # print right

        return cords_1, cords_2



    ##################################################
    ########## Plotting / Visualization  #############
    ##################################################


    def plot_3way_match( self, curr_im, pts_curr, prev_im, pts_prev, curr_m_im, pts_curr_m, enable_text=True, enable_lines=False, show_random_points=-1 ):
        """     pts_curr, pts_prev, pts_curr_m : 2xN numpy matrix

                returns : # grid : [ [curr, prev], [curr-1  X ] ]
        """

        print 'pts_curr.shape   ', pts_curr.shape
        print 'pts_prev.shape   ', pts_prev.shape
        print 'pts_curr_m.shape ', pts_curr_m.shape
        assert( (pts_curr.shape[1] == pts_prev.shape[1]) and (pts_curr_m.shape[1] == pts_prev.shape[1]) )
        assert( (pts_curr.shape[0] == pts_prev.shape[0]) and (pts_curr_m.shape[0] == pts_prev.shape[0]) )
        assert( pts_curr.shape[0] == 2 )
        # all 3 should have same number of points


        zero_image = np.zeros( curr_im.shape, dtype='uint8' )
        cv2.putText( zero_image, str(pts_curr.shape[1]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


        r1 = np.concatenate( ( curr_im, prev_im ), axis=1 )
        r2 = np.concatenate( ( curr_m_im, zero_image ), axis=1 )
        gridd = np.concatenate( (r1,r2), axis=0 )

        N = pts_curr.shape[1]
        if show_random_points < 0 or show_random_points > N:
            spann = range(N)
        else:
            spann = np.random.randint( 0, N, show_random_points )

        for xi in spann:
            point_c  = tuple( np.int0(pts_curr[:,xi]) )
            point_p  = tuple( np.int0(pts_prev[:,xi]) )
            point_cm = tuple( np.int0(pts_curr_m[:,xi]) )


            point_p__  = tuple( np.int0(pts_prev[:,xi]) + [curr_im.shape[1],0] )
            point_cm__ = tuple( np.int0(pts_curr_m[:,xi]) + [0,curr_im.shape[0]] )


            ######## C --- P
            cv2.circle( gridd, point_c, 4, (0,255,0) )
            cv2.circle( gridd, point_p__, 4, (0,255,0) )
            if enable_lines:
                cv2.line( gridd, point_c, point_p__, (255,0,0) )
            if enable_text:
                cv2.putText(gridd, str(xi), point_c, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )
                cv2.putText(gridd, str(xi), point_p__, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )


            ####### C --- Cm
            cv2.circle( gridd, point_cm__, 4, (0,255,0) )
            if enable_lines:
                cv2.line( gridd, point_c, point_cm__, (255,30,255) )
            if enable_text:
                cv2.putText(gridd, str(xi), point_cm__, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )




        return gridd


    def plot_point_sets( self, im1, pt1, im2, pt2, mask=None, enable_text=True, enable_lines=False ):
        """ pt1, pt2 : 2xN array """
        assert( pt1.shape[0] == 2 )
        assert( pt2.shape[0] == 2 )
        assert( pt1.shape[1] == pt2.shape[1] )

        #TODO: if im1 and im2 are 2 channel, this might cause issues

        xcanvas = np.concatenate( (im1, im2), axis=1 )
        for xi in range( pt1.shape[1] ):
            if (mask is not None) and (mask[xi,0] == 0):
                continue

            point_left =  tuple( np.int0(pt1[:,xi])  )
            point_right = tuple(np.int0(pt2[:,xi])   )
            point_right__ = tuple(np.int0(pt2[:,xi]) + [im1.shape[1],0])
            cv2.circle( xcanvas, point_left, 4, (255,0,255) )
            cv2.circle( xcanvas, point_right__, 4, (255,0,255) )

            if enable_lines:
                cv2.line( xcanvas, point_left, point_right__, (255,0,0) )

            if enable_text:
                cv2.putText(xcanvas, str(xi), point_left, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )
                cv2.putText(xcanvas, str(xi), point_right__, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )


        return xcanvas

    def plot_points_on_image( self, img, pts, enable_text=True ):
        """ img : Image; pts: 2xN array """

        xcanvas = img.copy()
        for xi in range( pts.shape[1] ):
            point_ =  tuple( np.int0(pts[:,xi])  )
            cv2.circle( xcanvas, point_, 4, (255,0,255) )

            if enable_text:
                cv2.putText(xcanvas, str(xi), point_, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0) )

        return xcanvas






def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))


if __name__ == '__main__':
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



if __name__ == '__xmain__':

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




if __name__ == '__xmain__':
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



if __name__ == '__xmain__':
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
