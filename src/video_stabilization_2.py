import sys
import os
import glob
import re
import numpy as np
import cv2 as cv
import argparse
import matplotlib.pyplot as plt


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

  
def read_frames_from_dir(directory):
  if not os.path.exists(directory):
    print ("Error directory does not exist")
    exit()
  # frames = [cv.imread(file) for file in sorted(glob.glob(directory + "/*.jpg"))]
  frames = []
  count = 0
  filenames = glob.glob(directory+ "/*.jpg")
  filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
  for file in filenames:
    if count >= 300:
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
    M, _ = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC)
    # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print(M)
    if M is None:
      return timewise_homographies, i
    timewise_homographies.append(np.transpose(M))
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
    pass


def apply_smoothing(original_frames, smooth_path, crop_ratio=0.8):
  print("smoothing new frames...")

  n = len(original_frames)
  w, h, _ = original_frames[0].shape
  centre_x, centre_y = (w/2, h/2)
  displacement_x, displacement_y = (crop_ratio * w)/2, (crop_ratio * h)/2

  top_left = np.array([centre_x - displacement_x, centre_y - displacement_y, 1])
  top_right = np.array([centre_x + displacement_x, centre_y - displacement_y, 1])
  bottom_left = np.array([centre_x - displacement_x, centre_y + displacement_y, 1])
  bottom_right = np.array([centre_x + displacement_x, centre_y + displacement_y, 1])
  crop_corners = [top_left, top_right, bottom_left, bottom_right]
  
  dst_corners = np.array([np.array([0, 0]), np.array([w, 0]), np.array([0, h]), np.array([w, h])]).astype(np.float32)

  new_frames = []
  for i in range(1, n):
    projected_corners = []
    for corner in crop_corners:
      projected_corners.append(corner.dot(smooth_path[i-1]))
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


def plot_paths(timewise_homographies, smooth_path):
    pass


def main():
  setup_folders()
  original_frames = read_frames_from_dir(sys.argv[1])
  features = extract_features(original_frames)
  timewise_homographies, _ = compute_timewise_homographies(original_frames, features, True)
  # smooth_path = compute_smooth_path(original_frames[0].shape, timewise_homographies)
  apply_smoothing(original_frames, timewise_homographies)
  # plot_paths(timewise_homographies, smooth_path)
  

if __name__ == "__main__":
  main()
