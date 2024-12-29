import argparse
import cv2
import numpy as np
import os
import sys
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable

# '''

# Sets up the directory folder structure/ensures that there is a
# data folder which contains the output folder a frames folder
# from which to extract/read frames from and a matches folder for which
# framewise pair matches are placed into for debugging

# '''
# def setup_folders():
#   try:
#     if not os.path.exists('data'):
#       os.makedirs('data')
#     if not os.path.exists('data/frames'):
#       os.makedirs('data/frames')
#     if not os.path.exists('data/output'):
#       os.makedirs('data/output')
#     if not os.path.exists('data/matches'):
#       os.makedirs('data/matches')
#   except OSError:
#       print ('Error: Creating directory')


# '''

# Given the path to a video extracts the frames from the video to the
# 'data/frame' folder and returns a list containing all the image frames

# '''
# def extract_frames_from_video(filename):
#   video = cv2.VideoCapture(filename)
#   frames = []
#   currentframe = 0
#   while(True):
#     ret,frame = video.read()
#     if ret:
#       frames.append(frame)
#       frame_filename = './data/frames/frame' + str(currentframe) + '.jpg'
#       print ('Extracting frame...' + frame_filename)
#       cv2.imwrite(frame_filename, frame)
#       currentframe += 1
#     else:
#       break
#   return frames


# '''

# Given a directory reads the images in ascending order of filename and
# returns the sequence of images in a list. Saves on having to extract frames
# from the video each time

# '''
# def read_frames_from_dir(directory):
#   if not os.path.exists(directory):
#     print ("Error directory does not exist")
#     exit()
#   frames = []
#   count = 0
#   filenames = glob.glob(directory+ "/*.jpg")
#   filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
#   for file in filenames:
#     # the following can be uncommmented to limit the number of frames read
#     if count >= 300:
#       break
#     frames.append(cv2.imread(file))
#     count+=1
#   print ("read " + str(len(frames)) + " frames from directory")
#   return frames


# '''

# Given a list of images which are frames of a video extracts the features
# from each and returns a list of tuples containing the (keypoint, descriptor).
# The feature detection algorithm used is ORB

# '''
# def extract_features(frames):
#   print ("extracting features...")
#   features = []
#   for i in range(len(frames)):
#     orb = cv2.ORB_create()
#     gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#     kp, des = orb.detectAndCompute(gray, None)
#     while np.array(kp).shape[0] == 0:
#       orb_less_accurate = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
#       print("frame: " + str(i) + " has no keypoints")
#       kp, des = orb_less_accurate.detectAndCompute(gray, None)
#     features.append((kp, des))
#   return features


# '''

# Computes the motion between each frame and returning back a list of affine transforms
# which describe the trajectory. Given n frames this method returns a list of n-1 transforms.

# '''
# def compute_timewise_homographies(frames, features, outputMatches=False):
#   print ("finding matches between frames...")
#   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
#   timewise_homographies = []
#   for i in range(len(frames) - 1):
#     matches = bf.match(features[i][1], features[i+1][1])
#     if outputMatches:
#       img3 = cv2.drawMatches(frames[i], features[i][0], frames[i+1], features[i+1][0], matches, None, flags=2)
#       cv2.imwrite('data/matches/matches_' + str(i) + "-" + str(i+1) + ".jpg", img3)
#     src_pts = np.float32([ features[i][0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
#     dst_pts = np.float32([ features[i+1][0][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
#     M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
#     if M is None:
#       return timewise_homographies, i
#     H = np.append(M, np.array([0, 0, 1]).reshape((1,3)), axis=0)
#     timewise_homographies.append(H)
#   return timewise_homographies, len(frames) - 1


# '''

# Helper method to return a list of corner points with a given crop ratio from the centre
# of the frame dimensions

# '''
# def get_corner_crop_pts(frame_dimensions, crop_ratio=0.8):
#   h, w, _ = frame_dimensions
#   centre_x, centre_y = (w/2, h/2)
#   displacement_x, displacement_y = (crop_ratio * w)/2, (crop_ratio * h)/2
#   top_left = (centre_x - displacement_x, centre_y - displacement_y)
#   top_right = (centre_x + displacement_x, centre_y - displacement_y)
#   bottom_left = (centre_x - displacement_x, centre_y + displacement_y)
#   bottom_right = (centre_x + displacement_x, centre_y + displacement_y)
#   corners = [top_left, top_right, bottom_left, bottom_right]
#   return corners


# '''

# Given the original trajectory of the camera and a crop ratio returns a smooth trajectory
# by using the linear programming technique described in the paper Auto-Directed Video
# Stabilization with Robust L1 Optimal Camera Paths
# (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37041.pdf)

# '''
# def compute_smooth_path(frame_dimensions, timewise_homographies=[], crop_ratio=0.8):
#   print("computing smooth path...")

#   n = len(timewise_homographies)

#   #
#   # (dx, dy, a, b, c, d)  <- form for the smooth_path vector
#   #
#   # | a   b   dx  |       | a   c   0  |
#   # | c   d   dy  |   ->  | b   d   0  |
#   # | 0   0   1   |   T   | dx  dy  1  |
#   #

#   weight_constant = 10                                          # weight towards constant 0 velocity path
#   weight_linear = 1                                             # weight towards segments with a constant non-zero velocity
#   weight_parabolic =  100                                       # weight towards segments with parabolic motion
#   affine_weights = np.transpose([1, 1, 100, 100, 100, 100])     # weighting of each component in the path vector we want to weight the affine portion more than the translation components
#   smooth_path = cp.Variable((n, 6))                             # matrix of the n smooth paths vectors that we are optimising to find
#   slack_var_1 = cp.Variable((n, 6))                             # Slack variable for constraining residual 1
#   slack_var_2 = cp.Variable((n, 6))                             # Slack variable for constraining residual 2
#   slack_var_3 = cp.Variable((n, 6))                             # Slack variable for constraining residual 3


#   # We define our optimisation as c^T @ e
#   objective = cp.Minimize(cp.sum((weight_constant * (slack_var_1 @ affine_weights)) +
#                                   (weight_linear * (slack_var_2 @ affine_weights)) +
#                                   (weight_parabolic * (slack_var_3 @ affine_weights)), axis=0))
#   constraints = []

#   # proximity constriants
#   # U is used to extract components from the vector smooth_path. We want to constrain
#   # the values of our path vector to the following:
#   # 0.9 <= a, d <= 1.1
#   # -0.1 <= b, c <= 0.1
#   # -0.1 <= a - d <= 0.1
#   # -0.051 <= b + c <= 0.05
#   U = np.array([0, 0, 0, 0, 0, 0,
#                 0, 0, 0, 0, 0, 0,
#                 1, 0, 0, 0, 1, 0,
#                 0, 1, 0, 0, 0, 1,
#                 0, 0, 1, 0, 0, 1,
#                 0, 0, 0, 1, -1, 0]).reshape(6, 6)
#   lb = np.array([0.9, -0.1, -0.1, 0.9, -0.1, -0.05])
#   ub = np.array([1.1, 0.1, 0.1, 1.1, 0.1, 0.05])
#   proximity = smooth_path @ U
#   for i in range(n):
#     constraints.append(proximity[i, :] >= lb)
#     constraints.append(proximity[i, :] <= ub)

#   # inclusion constraints for the crop corners
#   # want to make sure the corner points when projected are within the frame dimensions
#   corners = get_corner_crop_pts(frame_dimensions)
#   for corner in corners:
#     x, y = corner
#     projected_x = smooth_path @ np.transpose([1, 0, x, y, 0, 0])
#     projected_y = smooth_path @ np.transpose([0, 1, 0, 0, x, y])
#     constraints.append(projected_x >= 0)
#     constraints.append(projected_y >= 0)
#     constraints.append(projected_x <= frame_dimensions[1])
#     constraints.append(projected_y <= frame_dimensions[0])

#   # Smoothness constraints
#   constraints.append(slack_var_1 >= 0)
#   constraints.append(slack_var_2 >= 0)
#   constraints.append(slack_var_3 >= 0)

#   for i in range(n - 3):
#     # Extract smooth path component variables into a matrix we can then use to calculate each residual
#     # Residual 1 is for the constant zero velocity path
#     # Residual 2 is for the constant non-zero velocity path
#     # Residual 3 is for the parabolic non zero acceleration path
#     B_t = np.array([smooth_path[i, 2], smooth_path[i, 4], 0, smooth_path[i, 3], smooth_path[i, 5], 0, smooth_path[i, 0], smooth_path[i, 1], 1]).reshape((3,3))
#     B_t1 = np.array([smooth_path[i+1, 2], smooth_path[i+1, 4], 0, smooth_path[i+1, 3], smooth_path[i+1, 5], 0, smooth_path[i+1, 0], smooth_path[i+1, 1], 1]).reshape((3,3))
#     B_t2 = np.array([smooth_path[i+2, 2], smooth_path[i+2, 4], 0, smooth_path[i+2, 3], smooth_path[i+2, 5], 0, smooth_path[i+2, 0], smooth_path[i+2, 1], 1]).reshape((3,3))
#     B_t3 = np.array([smooth_path[i+3, 2], smooth_path[i+3, 4], 0, smooth_path[i+3, 3], smooth_path[i+3, 5], 0, smooth_path[i+3, 0], smooth_path[i+3, 1], 1]).reshape((3,3))

#     residual_t = np.transpose(timewise_homographies[i + 1]) @ B_t1  - B_t
#     residual_t1 = np.transpose(timewise_homographies[i + 2]) @ B_t2 - B_t1
#     residual_t2 = np.transpose(timewise_homographies[i + 3]) @ B_t3 - B_t2
#     residual_t = np.array([residual_t[2, 0], residual_t[2, 1], residual_t[0, 0], residual_t[1, 0], residual_t[0, 1], residual_t[1, 1]])
#     residual_t1 = np.array([residual_t1[2, 0], residual_t1[2, 1], residual_t1[0, 0], residual_t1[1, 0], residual_t1[0, 1], residual_t1[1, 1]])
#     residual_t2 = np.array([residual_t2[2, 0], residual_t2[2, 1], residual_t2[0, 0], residual_t2[1, 0], residual_t2[0, 1], residual_t2[1, 1]])

#     # this is where the actual smoothness constraint is obtained from the residuals
#     # i.e. this is where we summarized the following:
#     #  -e_t1 <= R_t(p) < e_t1
#     #  -e_t2 <= R_t1(p) - R_t(p) < e_t2
#     #  -e_t3 <= R_t2(p) - 2R_t1(p) + R_t(p) < e_t3
#     # if we can vectorize the below constraints we can speed this up and most likely get better results
#     # being able to smooth over more frames
#     for j in range(6):
#       constraints.append(residual_t[j] <= slack_var_1[i, j])
#       constraints.append(residual_t[j] >= -slack_var_1[i, j])
#       constraints.append((residual_t1[j] - residual_t[j]) <= slack_var_2[i, j])
#       constraints.append((residual_t1[j] - residual_t[j]) >= -slack_var_2[i, j])
#       constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) <= slack_var_3[i, j])
#       constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) >= -slack_var_3[i, j])

#   for i in range(n-3, n):
#     constraints.append(smooth_path[i, 5] == smooth_path[n-1, 5])

#   problem = cp.Problem(objective, constraints)
#   problem.solve(parallel=True, verbose=True)
#   print("status:", problem.status)
#   print("optimal value", problem.value)
#   print(smooth_path.value)
#   return convert_path_to_homography(smooth_path.value, n)


# '''

# converts each vector in the list path to a matrix of the form below:
#                               | a   b   dx  |
#   |dx dy a b c d|    to ->    | c   d   dy  |
#                               | 0   0   1   |
# and returns the list of numpy matrices

# '''
# def convert_path_to_homography(path, n):
#   smooth_homography = []
#   for i in range(n):
#     smooth_homography.append(np.array([path[i, 2], path[i, 3], path[i, 0],
#                                         path[i, 4], path[i, 5], path[i, 1],
#                                         0, 0, 1]).reshape(3, 3))
#   return smooth_homography


# '''

# Applies the smooth path the original frames cropping them to the
# given crop ratio. Writes the new frames to the subdirectory
# 'data/output/'

# '''
# def apply_smoothing(original_frames, smooth_path, crop_ratio=0.8):
#   print("smoothing new frames...")

#   n = len(original_frames)
#   h, w, _ = original_frames[0].shape
#   centre_x, centre_y = (w/2, h/2)
#   displacement_x, displacement_y = (crop_ratio * w)/2, (crop_ratio * h)/2

#   # get the displaced corner points from the centre of the frame which will become the
#   # corners of the resulting frame
#   top_left = np.array([centre_x - displacement_x, centre_y - displacement_y, 1])
#   top_right = np.array([centre_x + displacement_x, centre_y - displacement_y, 1])
#   bottom_left = np.array([centre_x - displacement_x, centre_y + displacement_y, 1])
#   bottom_right = np.array([centre_x + displacement_x, centre_y + displacement_y, 1])
#   crop_corners = [top_left, top_right, bottom_left, bottom_right]
#   dst_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]]).astype(np.float32)

#   # cycle through each frame projecting the corner to the smooth position and then
#   # find the resulting homography which can move the corners to the edges of the frame
#   # then warp the original frame according to this homography
#   new_frames = []
#   for i in range(n-1):
#     projected_corners = []
#     for corner in crop_corners:
#       projected_corners.append(smooth_path[i-1].dot(corner))
#     src_corners = np.array([[projected_corners[0][0], projected_corners[0][1]],
#                             [projected_corners[1][0], projected_corners[1][1]],
#                             [projected_corners[2][0], projected_corners[2][1]],
#                             [projected_corners[3][0], projected_corners[3][1]]]).astype(np.float32)
#     # H, _ = cv2.findHomography(src_corners, dst_corners, cv2.RANSAC, 5.0)
#     warp_frame = cv2.warpPerspective(original_frames[i], smooth_path[i], (original_frames[i].shape[1], original_frames[i].shape[0]))
#     frame_filename = './data/output/frame' + str(i) + '.jpg'
#     cv2.imwrite(frame_filename, warp_frame)
#     new_frames.append(warp_frame)
#   return new_frames


# '''

# Graphs the trajectory of the original and smooth paths and
# saves the graph as motion.png

# '''
# def graph_paths(timewise_homographies=[], smooth_path=[]):
#   print("graphing path...")
#   n = len(timewise_homographies)
#   original_x_path = np.zeros(n-1)
#   original_y_path = np.zeros(n-1)
#   original_dx = np.zeros(n-1)
#   original_dy = np.zeros(n-1)
#   smooth_x_path = np.zeros(n-1)
#   smooth_y_path = np.zeros(n-1)
#   smooth_dx_path = np.zeros(n-1)
#   smooth_dy_path = np.zeros(n-1)
#   pt = np.array([1, 1, 1])

#   # push point through the path and collect the result
#   for i in range(n-1):
#     pt = timewise_homographies[i].dot(pt)
#     original_x_path[i] = pt[0]
#     original_y_path[i] = pt[1]
#     # original_dx[i] = timewise_homographies[i][0,2]
#     # original_dy[i] = timewise_homographies[i][1,2]
#     smooth_pt = smooth_path[i].dot(pt)
#     smooth_x_path[i] = smooth_pt[0]
#     smooth_y_path[i] = smooth_pt[1]
#     # smooth_dx_path[i] = smooth_path[i][0,2]
#     # smooth_dy_path[i] = smooth_path[i][1,2]

#   # place data on the subplots
#   fig, axs = plt.subplots(1,2)
#   axs[0].set_title('x path')
#   axs[0].plot(np.arange(0, n-1), original_x_path, '-r')
#   axs[0].plot(np.arange(0, n-1), smooth_x_path, '-g')
#   axs[1].set_title('y path')
#   axs[1].plot(np.arange(0, n-1), original_y_path, '-r')
#   axs[1].plot(np.arange(0, n-1), smooth_y_path, '-g')
#   # axs[1, 0].set_title('dx path')
#   # axs[1, 0].plot(np.arange(0, n-1), original_dx, '-r')
#   # axs[1, 0].plot(np.arange(0, n-1), smooth_dx_path, '-g')
#   # axs[1, 1].set_title('dy path')
#   # axs[1, 1].plot(np.arange(0, n-1), original_dy, '-r')
#   # axs[1, 1].plot(np.arange(0, n-1), smooth_dy_path, '-g')
#   plt.savefig('motion.png')
#   plt.show()

N = 6  # num slack variables per residual 6 in the affine case, 8 for homography

# [constant weight (first derivative weight), linear weight (second derivative weight), parabolic weight (third derivative weight)]
w = [10, 1, 100]

# dx, dy, a, b, c, d
c = [1, 1, 100, 100, 100, 100, 100]


# returns the parameterized vector result of a
# matrix multiplication between F and B.
# F = [a b dx; c d dy]
# p = [p0, p1, p2, p3, p4, p5] corresponding to (dx, dy, a, b, c, d)
#
# F is a 2 x 3 matrix which we will be treated as an 3x3 augmented matrix:
# [a b dx]
# [c d dy]
# [0 0 1 ]
#
# B will be treated as a 3x3 augmented matrix constructed from p
# [p2 p3 p0]
# [p4 p5 p1]
# [0  0  1 ]
#
# The matrix result will be treated as F.T @ B.T
# [a * p2 + c * p3           a * p4 + c * p5           0]
# [b * p2 + d * p3           b * p4 + d * p5           0]
# [dx * p2 + dy * p3 + p0    dx * p4 + dy * p5 + p1    1]
#
# Transposing the above we get an affine projection which will
# be returned in the same format as how B is constructed
# of the input list
# [[2, 0], [2, 1], [0, 0], [1, 0], [0, 1], [1, 1]]
#    dx      dy      a       b       c       d
def get_parameterized_matmult(F, p):
    return [
        F[0, 2] * p[2] + F[1, 2] * p[3] + p[0],
        F[0, 2] * p[4] + F[1, 2] * p[5] + p[1],
        F[0, 0] * p[2] + F[1, 0] * p[3],
        F[0, 1] * p[2] + F[1, 1] * p[3],
        F[0, 0] * p[4] + F[1, 0] * p[5],
        F[0, 1] * p[4] + F[1, 1] * p[5]
    ]


class Stabilizer:
    def __init__(self, args):
        self.input_file = args.in_file
        self.output_file = args.out_file
        self.crop_ratio = args.crop_ratio
        self.plot = args.plot

        video = cv2.VideoCapture(self.input_file)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_slack_variables = N

    def run(self):
        transforms = self.estimate_per_frame_motion_transforms()
        update_transforms = self.compute_optimal_path(transforms)
        if not update_transforms:
            return
        self.stabilize_video(transforms, update_transforms)

    def stabilize_video(self, transforms, update_transforms):
        c_t = transforms[0].T
        augmented_b = np.hstack((update_transforms[100].T, np.zeros((3, 1))))
        smooth_transforms = [c_t @ augmented_b]
        for i in range(1, self.num_frames):
            augmented_f = np.hstack((transforms[i].T, np.zeros((3, 1))))
            c_t = c_t @ augmented_f
            augmented_b = np.hstack((update_transforms[i].T, np.zeros((3, 1))))
            smooth_transforms.append((c_t @ augmented_b).T)
            # print(smooth_transforms[-1])

    # returns a list of points of the four corners of the crop window bounds
    def get_crop_bounds(self):
        crop_w = round(self.width * self.crop_ratio)
        crop_h = round(self.height * self.crop_ratio)
        diff_w = self.width - crop_w
        diff_h = self.height - crop_h
        origin = (round(diff_w/2), round(diff_h/2))
        return [
            origin,
            (origin[0] + crop_w, origin[1]),
            (origin[0], origin[1] + crop_h),
            (origin[0] + crop_w, origin[1] + crop_h)
        ]

    # performs the linear programming optimization algorithm over the computed transforms
    # returns a list of transforms B which are the update transforms
    def compute_optimal_path(self, transforms):
        model = LpProblem(name="path_optimization", sense=LpMinimize)

        # parameterization vector for each frame (dx, dy, a, b, c, d)
        p = LpVariable.dicts("p", ((i, j)
                             for i in range(self.num_frames) for j in range(N)))

        # 3nN slack variables
        # e is a list per residual. There are N slack variables for each n frames
        e = []
        for i in range(3):
            e.append(LpVariable.dicts(f"e{i}", ((j, k) for j in range(self.num_frames)
                                                for k in range(N)), lowBound=0.0))

        # minimize (c^t)*e
        # O(P) = w1|D(P)| + w2|D2(P)| + w3|D3(P)|
        optimization_objective = 0
        for i in range(3):
            optimization_objective += w[i] * lpSum(e[i][j, k] * c[k]
                                                   for j in range(self.num_frames) for k in range(N))
        model += optimization_objective

        crop_bounds = self.get_crop_bounds()
        for i in range(self.num_frames):
            if i < self.num_frames - 3:
                # smoothness constraints
                # −e1_t ≤ R_t(p) ≤ e1_t
                # −e2_t ≤ R_t+1(p) − R_t(p) ≤ e2_t
                # −e3_t ≤ R_t+2(p) − 2R_t+1(p) + R_t(p) ≤ e3_t
                # ei_t ≥ 0 <- satisfied by lowBound on LpVariable
                #
                # R = F_(t+1) * B_(t+1) - p_t
                Mt = get_parameterized_matmult(
                    transforms[i+1], [p[i+1, j] for j in range(N)])
                Mt1 = get_parameterized_matmult(
                    transforms[i+2], [p[i+2, j] for j in range(N)])
                Mt2 = get_parameterized_matmult(
                    transforms[i+3], [p[i+3, j] for j in range(N)])

                r_t = [Mt[j] - p[i, j] for j in range(N)]
                r_t1 = [Mt1[j] - p[i + 1, j] for j in range(N)]
                r_t2 = [Mt2[j] - p[i + 2, j] for j in range(N)]

                for j in range(N):
                    model += -1 * e[0][i, j] <= r_t[j]
                    model += r_t[j] <= e[0][i, j]

                    model += -1 * e[1][i, j] <= r_t1[j] - r_t[j]
                    model += r_t1[j] - r_t[j] <= e[1][i, j]

                    model += -1 * e[2][i, j] <= r_t2[j] - 2 * r_t1[j] + r_t[j]
                    model += r_t2[j] - 2 * r_t1[j] + r_t[j] <= e[2][i, j]

            # proximity constraints
            # lb ≤ Upt ≤ ub
            # 0.9 ≤ at,dt ≤ 1.1,
            # −0.1 ≤ bt,ct ≤ 0.1
            # −0.05 ≤ bc + ct ≤ 0.05
            # −0.1 ≤ at − dt ≤0.1
            # The first two constraints limit the range of change in
            # zoom and rotation, while the latter two give the affine trans-
            # form more rigidity by limiting the amount of skew and non-
            # uniform scale.
            model += 0.9 <= p[i, 2] <= 1.1
            model += p[i, 5] >= 0.9
            model += p[i, 5] <= 1.1
            model += -0.1 <= p[i, 3] <= 0.1
            model += -0.1 <= p[i, 4] <= 0.1
            model += -0.1 <= p[i, 3] + p[i, 4] <= 0.1
            model += -0.05 <= p[i, 2] - p[i, 5] <= 0.05

            # Inclusion Constraints
            # (0, 0)T ≤ CR_i * pt ≤ (w, h)T
            for x, y in crop_bounds:
                model += 0 <= p[i, 0] + p[i, 2] * x + p[i, 3] * y <= self.width
                model += 0 <= p[i, 1] + p[i, 4] * \
                    x + p[i, 5] * y <= self.height

        print("Running solver")
        model.solve()
        smooth_transforms = []
        if model.status == 1:
            print("solution found")
            for i in range(self.num_frames):
                smooth_transforms.append(
                    np.array([[p[i, 2].varValue, p[i, 4].varValue, p[i, 0].varValue],
                              [p[i, 3].varValue, p[i, 5].varValue, p[i, 1].varValue]])
                )
        else:
            print("unable to find solution")
            return None

        return smooth_transforms

    # computes the linear motion model (affine transform homography) between each pair of frames and returns each in a list
    def estimate_per_frame_motion_transforms(self):
        video = cv2.VideoCapture(self.input_file)

        # first transform is just the identity matrix
        transforms = [np.eye(3, 3)]

        feature_params = dict(maxCorners=200,
                              qualityLevel=0.01,
                              minDistance=30,
                              blockSize=3)
        lk_params = dict(winSize=(20, 20),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # pairwise through the video frames compute the optical flow and compute the linear motion
        # model F_t(x) between I_t-1 and I_t
        success, prev_frame = video.read()
        if not success:
            print("unable to extract first frame")
            sys.exit()
        prev_frame_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for i in range(self.num_frames - 1):
            prev_feature_points = cv2.goodFeaturesToTrack(
                prev_frame_grey, mask=None, **feature_params)
            success, curr_frame = video.read()
            if not success:
                print(f"failed to read frame {i} aborting")
                sys.exit()
            curr_frame_grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            curr_feature_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame_grey, curr_frame_grey, prev_feature_points, None)

            if curr_feature_points is None:
                print(
                    "unable to perform optical flow for frame stopping at frame {i}/{self.num_frames}")
                break

            # grab the features which have been successfully tracked by optical flow
            valid_curr_feature_points = curr_feature_points[status == 1]
            valid_prev_feature_points = prev_feature_points[status == 1]

            if len(valid_curr_feature_points) < 4:
                print(
                    f"cannot compute homography between frames that have less than three features stopping at frame {i}/{self.num_frames}")
                break

            # compute the affine transform between matched prev and curr features
            M, _ = cv2.estimateAffine2D(
                valid_curr_feature_points, valid_prev_feature_points, method=cv2.RANSAC)
            transforms.append(M)
            prev_frame_grey = curr_frame_grey.copy()
        self.num_frames = len(transforms)
        return transforms


def setup_folders():
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('data/frames'):
            os.makedirs('data/frames')
        if not os.path.exists('data/output'):
            os.makedirs('data/output')
        if not os.path.exists('data/matches'):
            os.makedirs('data/matches')
    except OSError:
        print('Error: Creating directory')


def parse_args():
    parser = argparse.ArgumentParser(prog='L1 Video Stabilizer')
    parser.add_argument(
        'in_file',
        type=str,
        help='input video file to stabilize')
    parser.add_argument(
        '-o',
        '--out_file',
        type=str,
        help="name of the output file default will be the input file name with the suffix _stable")
    parser.add_argument(
        '-c',
        '--crop_ratio',
        type=float,
        default=0.7,
        help='crop ratio for the crop window [0, 1]')
    parser.add_argument(
        '-p',
        '--plot',
        action='store_true',
        help='flag for whether to save and output the plot graphs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_folders()
    stabilizer = Stabilizer(args)
    stabilizer.run()
