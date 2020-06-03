import sys
import os
import glob
import re
import math
import cvxpy as cp
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''

Sets up the directory folder structure/ensures that there is a 
data folder which contains the output folder a frames folder 
from which to extract/read frames from and a matches folder for which 
framewise pair matches are placed into for debugging

'''
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
      print ('Error: Creating directory') 


'''

Given the path to a video extracts the frames from the video to the 
'data/frame' folder and returns a list containing all the image frames

'''
def extract_frames_from_video(filename):
  video = cv.VideoCapture(filename)
  frames = []
  currentframe = 0
  while(True): 
    ret,frame = video.read() 
    if ret: 
      frames.append(frame)
      frame_filename = './data/frames/frame' + str(currentframe) + '.jpg'
      print ('Extracting frame...' + frame_filename) 
      cv.imwrite(frame_filename, frame)
      currentframe += 1
    else: 
      break
  return frames

  
'''

Given a directory reads the images in ascending order of filename and
returns the sequence of images in a list. Saves on having to extract frames
from the video each time

'''
def read_frames_from_dir(directory):
  if not os.path.exists(directory):
    print ("Error directory does not exist")
    exit()
  frames = []
  count = 0
  filenames = glob.glob(directory+ "/*.jpg")
  filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
  for file in filenames:
    # the following can be uncommmented to limit the number of frames read
    # if count >= 300:
    #   break
    frames.append(cv.imread(file))
    count+=1
  print ("read " + str(len(frames)) + " frames from directory")
  return frames


'''

Given a list of images which are frames of a video extracts the features 
from each and returns a list of tuples containing the (keypoint, descriptor).
The feature detection algorithm used is ORB

'''
def extract_features(frames):
  print ("extracting features...")
  features = []
  for i in range(len(frames)):
    orb = cv.ORB_create()
    gray = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    while np.array(kp).shape[0] == 0:
      orb_less_accurate = cv.ORB_create(nfeatures=1000, scoreType=cv.ORB_FAST_SCORE)
      print("frame: " + str(i) + " has no keypoints")
      kp, des = orb_less_accurate.detectAndCompute(gray, None)
    features.append((kp, des))
  return features


'''

Computes the motion between each frame and returning back a list of affine transforms
which describe the trajectory. Given n frames this method returns a list of n-1 transforms.

'''
def compute_timewise_homographies(frames, features, outputMatches=False):
  print ("finding matches between frames...")
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
  timewise_homographies = []
  for i in range(len(frames) - 1):
    matches = bf.match(features[i][1], features[i+1][1])
    if outputMatches:
      img3 = cv.drawMatches(frames[i], features[i][0], frames[i+1], features[i+1][0], matches, None, flags=2)
      cv.imwrite('data/matches/matches_' + str(i) + "-" + str(i+1) + ".jpg", img3)
    src_pts = np.float32([ features[i][0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ features[i+1][0][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, _ = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC)
    if M is None:
      return timewise_homographies, i
    H = np.append(M, np.array([0, 0, 1]).reshape((1,3)), axis=0)
    timewise_homographies.append(H)
  return timewise_homographies, len(frames) - 1


'''

Helper method to return a list of corner points with a given crop ratio from the centre 
of the frame dimensions

'''
def get_corner_crop_pts(frame_dimensions, crop_ratio=0.8):
  h, w, _ = frame_dimensions
  centre_x, centre_y = (w/2, h/2)
  displacement_x, displacement_y = (crop_ratio * w)/2, (crop_ratio * h)/2
  top_left = (centre_x - displacement_x, centre_y - displacement_y)
  top_right = (centre_x + displacement_x, centre_y - displacement_y)
  bottom_left = (centre_x - displacement_x, centre_y + displacement_y)
  bottom_right = (centre_x + displacement_x, centre_y + displacement_y)
  corners = [top_left, top_right, bottom_left, bottom_right]
  return corners



'''

Given the original trajectory of the camera and a crop ratio returns a smooth trajectory
by using the linear programming technique described in the paper Auto-Directed Video 
Stabilization with Robust L1 Optimal Camera Paths
(https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37041.pdf)

'''
def compute_smooth_path(frame_dimensions, timewise_homographies=[], crop_ratio=0.8):
  print("computing smooth path...")

  n = len(timewise_homographies)

  # 
  # (dx, dy, a, b, c, d)  <- form for the smooth_path vector
  # 
  # | a   b   dx  |       | a   c   0  |
  # | c   d   dy  |   ->  | b   d   0  |
  # | 0   0   1   |   T   | dx  dy  1  |
  #

  weight_constant = 10                                          # weight towards constant 0 velocity path
  weight_linear = 1                                             # weight towards segments with a constant non-zero velocity 
  weight_parabolic =  100                                       # weight towards segments with parabolic motion
  affine_weights = np.transpose([1, 1, 100, 100, 100, 100])     # weighting of each component in the path vector we want to weight the affine portion more than the translation components
  smooth_path = cp.Variable((n, 6))                             # matrix of the n smooth paths vectors that we are optimising to find 
  slack_var_1 = cp.Variable((n, 6))                             # Slack variable for constraining residual 1
  slack_var_2 = cp.Variable((n, 6))                             # Slack variable for constraining residual 2
  slack_var_3 = cp.Variable((n, 6))                             # Slack variable for constraining residual 3


  # We define our optimisation as c^T @ e 
  objective = cp.Minimize(cp.sum((weight_constant * (slack_var_1 @ affine_weights)) +
                                  (weight_linear * (slack_var_2 @ affine_weights)) +
                                  (weight_parabolic * (slack_var_3 @ affine_weights)), axis=0))
  constraints = []

  # proximity constriants
  # U is used to extract components from the vector smooth_path. We want to constrain 
  # the values of our path vector to the following: 
  # 0.9 <= a, d <= 1.1
  # -0.1 <= b, c <= 0.1
  # -0.1 <= a - d <= 0.1
  # -0.051 <= b + c <= 0.05
  U = np.array([0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 1, 0,
                0, 1, 0, 0, 0, 1,
                0, 0, 1, 0, 0, 1,
                0, 0, 0, 1, -1, 0]).reshape(6, 6)
  lb = np.array([0.9, -0.1, -0.1, 0.9, -0.1, -0.05])
  ub = np.array([1.1, 0.1, 0.1, 1.1, 0.1, 0.05])
  proximity = smooth_path @ U
  for i in range(n):
    constraints.append(proximity[i, :] >= lb)
    constraints.append(proximity[i, :] <= ub)
  
  # inclusion constraints for the crop corners
  # want to make sure the corner points when projected are within the frame dimensions
  corners = get_corner_crop_pts(frame_dimensions)
  for corner in corners:
    x, y = corner
    projected_x = smooth_path @ np.transpose([1, 0, x, y, 0, 0])
    projected_y = smooth_path @ np.transpose([0, 1, 0, 0, x, y])
    constraints.append(projected_x >= 0)
    constraints.append(projected_y >= 0)
    constraints.append(projected_x <= frame_dimensions[1])
    constraints.append(projected_y <= frame_dimensions[0])
  
  # Smoothness constraints
  constraints.append(slack_var_1 >= 0)
  constraints.append(slack_var_2 >= 0)
  constraints.append(slack_var_3 >= 0)

  for i in range(n - 3):
    # Extract smooth path component variables into a matrix we can then use to calculate each residual
    # Residual 1 is for the constant zero velocity path
    # Residual 2 is for the constant non-zero velocity path
    # Residual 3 is for the parabolic non zero acceleration path
    B_t = np.array([smooth_path[i, 2], smooth_path[i, 4], 0, smooth_path[i, 3], smooth_path[i, 5], 0, smooth_path[i, 0], smooth_path[i, 1], 1]).reshape((3,3))
    B_t1 = np.array([smooth_path[i+1, 2], smooth_path[i+1, 4], 0, smooth_path[i+1, 3], smooth_path[i+1, 5], 0, smooth_path[i+1, 0], smooth_path[i+1, 1], 1]).reshape((3,3))
    B_t2 = np.array([smooth_path[i+2, 2], smooth_path[i+2, 4], 0, smooth_path[i+2, 3], smooth_path[i+2, 5], 0, smooth_path[i+2, 0], smooth_path[i+2, 1], 1]).reshape((3,3))
    B_t3 = np.array([smooth_path[i+3, 2], smooth_path[i+3, 4], 0, smooth_path[i+3, 3], smooth_path[i+3, 5], 0, smooth_path[i+3, 0], smooth_path[i+3, 1], 1]).reshape((3,3))

    residual_t = np.transpose(timewise_homographies[i + 1]) @ B_t1  - B_t
    residual_t1 = np.transpose(timewise_homographies[i + 2]) @ B_t2 - B_t1
    residual_t2 = np.transpose(timewise_homographies[i + 3]) @ B_t3 - B_t2
    residual_t = np.array([residual_t[2, 0], residual_t[2, 1], residual_t[0, 0], residual_t[1, 0], residual_t[0, 1], residual_t[1, 1]])
    residual_t1 = np.array([residual_t1[2, 0], residual_t1[2, 1], residual_t1[0, 0], residual_t1[1, 0], residual_t1[0, 1], residual_t1[1, 1]])
    residual_t2 = np.array([residual_t2[2, 0], residual_t2[2, 1], residual_t2[0, 0], residual_t2[1, 0], residual_t2[0, 1], residual_t2[1, 1]])

    # this is where the actual smoothness constraint is obtained from the residuals
    # i.e. this is where we summarized the following:
    #  -e_t1 <= R_t(p) < e_t1
    #  -e_t2 <= R_t1(p) - R_t(p) < e_t2
    #  -e_t3 <= R_t2(p) - 2R_t1(p) + R_t(p) < e_t3
    # if we can vectorize the below constraints we can speed this up and most likely get better results
    # being able to smooth over more frames
    for j in range(6):
      constraints.append(residual_t[j] <= slack_var_1[i, j])
      constraints.append(residual_t[j] >= -slack_var_1[i, j])
      constraints.append((residual_t1[j] - residual_t[j]) <= slack_var_2[i, j])
      constraints.append((residual_t1[j] - residual_t[j]) >= -slack_var_2[i, j])
      constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) <= slack_var_3[i, j])
      constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) >= -slack_var_3[i, j])
      
  for i in range(n-3, n):
    constraints.append(smooth_path[i, 5] == smooth_path[n-1, 5])
  
  problem = cp.Problem(objective, constraints)
  problem.solve(parallel=True, verbose=True)
  print("status:", problem.status)
  print("optimal value", problem.value)
  print(smooth_path.value)
  return convert_path_to_homography(smooth_path.value, n)


'''

converts each vector in the list path to a matrix of the form below:
                              | a   b   dx  | 
  |dx dy a b c d|    to ->    | c   d   dy  |
                              | 0   0   1   |
and returns the list of numpy matrices

'''
def convert_path_to_homography(path, n):
  smooth_homography = []
  for i in range(n):
    smooth_homography.append(np.array([path[i, 2], path[i, 3], path[i, 0], 
                                        path[i, 4], path[i, 5], path[i, 1], 
                                        0, 0, 1]).reshape(3, 3))
  return smooth_homography


'''

Applies the smooth path the original frames cropping them to the 
given crop ratio. Writes the new frames to the subdirectory 
'data/output/'

'''
def apply_smoothing(original_frames, smooth_path, crop_ratio=0.8):
  print("smoothing new frames...")

  n = len(original_frames)
  h, w, _ = original_frames[0].shape
  centre_x, centre_y = (w/2, h/2)
  displacement_x, displacement_y = (crop_ratio * w)/2, (crop_ratio * h)/2

  # get the displaced corner points from the centre of the frame which will become the 
  # corners of the resulting frame
  top_left = np.array([centre_x - displacement_x, centre_y - displacement_y, 1])
  top_right = np.array([centre_x + displacement_x, centre_y - displacement_y, 1])
  bottom_left = np.array([centre_x - displacement_x, centre_y + displacement_y, 1])
  bottom_right = np.array([centre_x + displacement_x, centre_y + displacement_y, 1])
  crop_corners = [top_left, top_right, bottom_left, bottom_right]
  dst_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]]).astype(np.float32)

  # cycle through each frame projecting the corner to the smooth position and then 
  # find the resulting homography which can move the corners to the edges of the frame
  # then warp the original frame according to this homography
  new_frames = []
  for i in range(n-1):
    projected_corners = []
    for corner in crop_corners:
      projected_corners.append(smooth_path[i-1].dot(corner))
    src_corners = np.array([[projected_corners[0][0], projected_corners[0][1]],
                            [projected_corners[1][0], projected_corners[1][1]], 
                            [projected_corners[2][0], projected_corners[2][1]],
                            [projected_corners[3][0], projected_corners[3][1]]]).astype(np.float32)
    H, _ = cv.findHomography(src_corners, dst_corners, cv.RANSAC, 5.0)
    warp_frame = cv.warpPerspective(original_frames[i], H, (original_frames[i].shape[1], original_frames[i].shape[0]))
    frame_filename = './data/output/frame' + str(i) + '.jpg'
    cv.imwrite(frame_filename, warp_frame)
    new_frames.append(warp_frame)
  return new_frames


'''

Graphs the trajectory of the original and smooth paths and 
saves the graph as motion.png

'''
def graph_paths(timewise_homographies=[], smooth_path=[]):
  print("graphing path...")
  n = len(timewise_homographies)
  original_x_path = np.zeros(n-1)
  original_y_path = np.zeros(n-1)
  original_dx = np.zeros(n-1)
  original_dy = np.zeros(n-1)
  smooth_x_path = np.zeros(n-1)
  smooth_y_path = np.zeros(n-1)
  smooth_dx_path = np.zeros(n-1)
  smooth_dy_path = np.zeros(n-1)
  pt = np.array([1, 1, 1])

  # push point through the path and collect the result
  for i in range(n-1):
    pt = timewise_homographies[i].dot(pt)
    original_x_path[i] = pt[0]
    original_y_path[i] = pt[1]
    original_dx[i] = timewise_homographies[i][0,2]
    original_dy[i] = timewise_homographies[i][1,2]
    smooth_pt = smooth_path[i].dot(pt)
    smooth_x_path[i] = smooth_pt[0]
    smooth_y_path[i] = smooth_pt[1]
    smooth_dx_path[i] = smooth_path[i][0,2]
    smooth_dy_path[i] = smooth_path[i][1,2]

  # place data on the subplots
  fig, axs = plt.subplots(2,2)
  axs[0, 0].set_title('x path')
  axs[0, 0].plot(np.arange(0, n-1), original_x_path, '-r')
  axs[0, 0].plot(np.arange(0, n-1), smooth_x_path, '-g')
  axs[0, 1].set_title('y path')
  axs[0, 1].plot(np.arange(0, n-1), original_y_path, '-r')
  axs[0, 1].plot(np.arange(0, n-1), smooth_y_path, '-g')
  axs[1, 0].set_title('dx path')
  axs[1, 0].plot(np.arange(0, n-1), original_dx, '-r')
  axs[1, 0].plot(np.arange(0, n-1), smooth_dx_path, '-g')
  axs[1, 1].set_title('dy path')
  axs[1, 1].plot(np.arange(0, n-1), original_dy, '-r')
  axs[1, 1].plot(np.arange(0, n-1), smooth_dy_path, '-g')
  plt.savefig('motion.png')
  plt.show()


def main():
  setup_folders()
  original_frames = read_frames_from_dir(sys.argv[1])
  features = extract_features(original_frames)
  timewise_homographies, _ = compute_timewise_homographies(original_frames, features)
  smooth_path = compute_smooth_path(original_frames[0].shape, timewise_homographies)
  apply_smoothing(original_frames, smooth_path)
  graph_paths(timewise_homographies, smooth_path)
  

if __name__ == "__main__":
  main()
