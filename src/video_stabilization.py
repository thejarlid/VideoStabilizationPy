import sys
import os
import glob
import cvxpy as cp
import numpy as np
import cv2 as cv
import argparse


def extract_frames_from_video(filename):
  video = cv.VideoCapture(filename)

  # creating a folder named data 
  try: 
    if not os.path.exists('data'): 
      os.makedirs('data')
    if not os.path.exists('data/frames'):
      os.makedirs('data/frames')
  except OSError: 
      print ('Error: Creating directory of data') 

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

  
def read_frames_from_dir(directory):
  if not os.path.exists(directory):
    print ("Error directory does not exist")
    exit()
  # frames = [cv.imread(file) for file in sorted(glob.glob(directory + "/*.jpg"))]
  frames = []
  count = 0
  for file in sorted(glob.glob(directory + "/*.jpg")):
    if count >= 3000:
      break
    frames.append(cv.imread(file))
    count+=1
  print ("read " + str(len(frames)) + " frames from directory")
  return frames


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


def compute_timewise_homographies(frames, features):
  print ("finding matches between frames...")
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
  timewise_homographies = []
  for i in range(len(frames) - 1):
    matches = bf.match(features[i][1], features[i+1][1])
    src_pts = np.float32([ features[i][0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ features[i+1][0][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    if M is None:
      return timewise_homographies, i
    timewise_homographies.append(np.array(M))
  return timewise_homographies, len(frames) - 1


def get_corner_crop_pts(frame_dimensions, crop_ratio=0.8):
  centre_pt_original = (round(frame_dimensions[1]/2), round(frame_dimensions[0]/2))     # (x, y)
  new_dimensions = (frame_dimensions[1] * crop_ratio, frame_dimensions[0] * crop_ratio) # (w x h)

  # crop corners
  top_left = (centre_pt_original[0] - new_dimensions[0]/2, centre_pt_original[1] - new_dimensions[1]/2)
  top_right = (centre_pt_original[0] + new_dimensions[0]/2, centre_pt_original[1] - new_dimensions[1]/2)
  bottom_left = (centre_pt_original[0] - new_dimensions[0]/2, centre_pt_original[1] + new_dimensions[1]/2)
  bottom_right = (centre_pt_original[0] + new_dimensions[0]/2, centre_pt_original[1] + new_dimensions[1]/2)
  corners = [top_left, top_right, bottom_left, bottom_right]
  return corners


def compute_smooth_path(frame_dimensions, timewise_homographies=[], crop_ratio=0.8):
  print("computing smooth path...")

  n = len(timewise_homographies)

  weight_constant = 10
  weight_linear = 1
  weight_parabolic =  100

  affine_weights_1 = np.transpose(np.array([1, 1, 100, 100, 100, 100]))
  affine_weights_2 = np.transpose(np.array([1, 1, 100, 100, 100, 100]))
  affine_weights_3 = np.transpose(np.array([1, 1, 100, 100, 100, 100]))

  smooth_path = cp.Variable((n, 6))
  slack_var_1 = cp.Variable((n, 6))
  slack_var_2 = cp.Variable((n, 6))
  slack_var_3 = cp.Variable((n, 6))

  objective = cp.Minimize(cp.sum((weight_constant * (slack_var_1 @ affine_weights_1)) +
                                  (weight_linear * (slack_var_2 @ affine_weights_2)) +
                                  (weight_parabolic * (slack_var_3 @ affine_weights_3)), axis=0))

  constraints = []

  # proximity constriants

  
  # inclusion constraints for the crop corners
  # centre_pt_original = (round(frame_dimensions[1]/2), round(frame_dimensions[0]/2))     # (x, y)
  # new_dimensions = (frame_dimensions[1] * crop_ratio, frame_dimensions[0] * crop_ratio) # (w x h)

  # # crop corners
  # top_left = (centre_pt_original[0] - new_dimensions[0]/2, centre_pt_original[1] - new_dimensions[1]/2)
  # top_right = (centre_pt_original[0] + new_dimensions[0]/2, centre_pt_original[1] - new_dimensions[1]/2)
  # bottom_left = (centre_pt_original[0] - new_dimensions[0]/2, centre_pt_original[1] + new_dimensions[1]/2)
  # bottom_right = (centre_pt_original[0] + new_dimensions[0]/2, centre_pt_original[1] + new_dimensions[1]/2)
  corners = get_corner_crop_pts(frame_dimensions)

  for corner in corners:
    for i in range(n):
        x, y = corner
        horizontal = np.array([1, 0, x, y, 0, 0])
        vertical = np.array([0, 1, 0, 0, x, y])
        constraints.append(horizontal @ smooth_path[i] >= 0)
        constraints.append(vertical @ smooth_path[i] >= 0)
        constraints.append(horizontal @ smooth_path[i] <= frame_dimensions[1])
        constraints.append(vertical @ smooth_path[i] <= frame_dimensions[0])
  
  # Smoothness constraints
  constraints.append(slack_var_1 >= 0)
  constraints.append(slack_var_2 >= 0)
  constraints.append(slack_var_3 >= 0)

  for i in range(n - 3):
    # format B_tx so that when multiplied by the timewise homography to get the residual the affine
    # transform is kept
    B_t = np.array([smooth_path[i, 2], smooth_path[i, 4], 0, 
                    smooth_path[i, 3], smooth_path[i, 5], 0, 
                    smooth_path[i, 0], smooth_path[i, 1], 1]).reshape((3,3))
    B_t1 = np.array([smooth_path[i+1, 2], smooth_path[i+1, 4], 0, 
                    smooth_path[i+1, 3], smooth_path[i+1, 5], 0, 
                    smooth_path[i+1, 0], smooth_path[i+1, 1], 1]).reshape((3,3))
    B_t2 = np.array([smooth_path[i+2, 2], smooth_path[i+2, 4], 0, 
                    smooth_path[i+2, 3], smooth_path[i+2, 5], 0, 
                    smooth_path[i+2, 0], smooth_path[i+2, 1], 1]).reshape((3,3))
    B_t3 = np.array([smooth_path[i+3, 2], smooth_path[i+3, 4], 0, 
                    smooth_path[i+3, 3], smooth_path[i+3, 5], 0, 
                    smooth_path[i+3, 0], smooth_path[i+3, 1], 1]).reshape((3,3))

    residual_t = np.array(timewise_homographies[i + 1]).dot(B_t1) - B_t
    residual_t1 = np.array(timewise_homographies[i + 2]).dot(B_t2) - B_t1
    residual_t2 = np.array(timewise_homographies[i + 3]).dot(B_t3) - B_t2
    residual_t = np.array([residual_t[2, 0], residual_t[2, 1], residual_t[0, 0], residual_t[1, 0], residual_t[0, 1], residual_t[1, 1]])
    residual_t1 = np.array([residual_t1[2, 0], residual_t1[2, 1], residual_t1[0, 0], residual_t1[1, 0], residual_t1[0, 1], residual_t1[1, 1]])
    residual_t2 = np.array([residual_t2[2, 0], residual_t2[2, 1], residual_t2[0, 0], residual_t2[1, 0], residual_t2[0, 1], residual_t2[1, 1]])

    # this is where the actual smoothness constraint is obtained from the residuals
    # i.e. this is where we summarized the following:
    #  -e_t1 <= R_t(p) < et1
    #  -e_t2 <= R_t1(p) - R_t(p) < et2
    #  -e_t3 <= R_t2(p) - 2R_t1(p) + R_t(p) < et3
    for j in range(6):
      constraints.append(residual_t[j] <= slack_var_1[i, j])
      constraints.append(residual_t[j] >= -slack_var_1[i, j])
      constraints.append((residual_t1[j] - residual_t[j]) <= slack_var_2[i, j])
      constraints.append((residual_t1[j] - residual_t[j]) >= -slack_var_2[i, j])
      constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) <= slack_var_3[i, j])
      constraints.append((residual_t2[j] - 2*residual_t1[j] + residual_t[j]) >= -slack_var_3[i, j])
    
  problem = cp.Problem(objective, constraints)
  problem.solve(parallel=True)
  print("status:", problem.status)
  print("optimal value", problem.value)
  print(smooth_path.value)
  return smooth_path


def smoot(original_frames, smooth_path, crop_ratio=0.8):
  pass


def main():
  original_frames = read_frames_from_dir(sys.argv[1])
  features = extract_features(original_frames)
  timewise_homographies, count = compute_timewise_homographies(original_frames, features)
  smooth_path = compute_smooth_path(original_frames[0].shape, timewise_homographies)


if __name__ == "__main__":
  main()
