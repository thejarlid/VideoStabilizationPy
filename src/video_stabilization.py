import argparse
import cv2
import numpy as np
import os
import sys
from pulp import LpMinimize, LpProblem, lpSum, LpVariable
import matplotlib.pyplot as plt

N = 6  # num slack variables per residual 6 in the affine case, 8 for homography

# [constant weight (first derivative weight), linear weight (second derivative weight), parabolic weight (third derivative weight)]
w = [10, 1, 100]

# dx, dy, a, b, c, d
c = [1, 1, 100, 100, 100, 100, 100]


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
        self.stabilize_video(update_transforms)
        if self.plot:
            self.plot_paths(transforms, update_transforms)

    def plot_paths(self, transforms, update_transforms):
        # the camera path C(t) of the original video footage is iteratively computed
        # by the matrix multiplication Ct+1 = Ct * Ft+1 = ⇒ Ct = F1 * F2 * ... * Ft.
        # Given the original path Ct, we express the desired smooth path as Pt = Ct * Bt
        # where Bt = C^-1 * Pt is the update transform that when applied to the original
        # camera path Ct, yields the optimal path Pt.
        c_t = np.vstack((transforms[0], [0, 0, 1])).T
        original_path = []
        smooth_path = []
        pt = np.array([0, 0, 1])
        for i in range(1, self.num_frames):
            c_t = c_t @ np.vstack((transforms[i], [0, 0, 1])).T
            p_t = c_t @ np.vstack((update_transforms[i], [0, 0, 1])).T
            original_path.append(pt @ c_t)
            smooth_path.append(pt @ p_t)
        _, axs = plt.subplots(1, 2, figsize=(10, 7))
        axs[0].plot([pt[0] for pt in original_path])
        axs[0].plot([pt[0] for pt in smooth_path])
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('x path')
        axs[0].legend(['Original', 'Smooth'])

        axs[1].plot([pt[1] for pt in original_path])
        axs[1].plot([pt[1] for pt in smooth_path])
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('y coord')
        axs[1].legend(['Original', 'Smooth'])
        plt.show()
        plt.savefig("path.png")
        plt.close()

    def stabilize_video(self, update_transforms):
        # create new video
        video = cv2.VideoCapture(self.input_file)
        out_video = cv2.VideoWriter(
            self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), int(video.get(cv2.CAP_PROP_FPS)), (self.width, self.height))

        # used to ensure that scaling, rotation, or other transformations are applied
        # relative to the center of the image, rather than the top-left corner (the origin
        # in image coordinates). This helps maintain visual coherence and prevents
        # unwanted distortions.
        scaling_matrix = np.eye(3, dtype=float)
        scaling_matrix[0][0] = 1/self.crop_ratio
        scaling_matrix[1][1] = 1/self.crop_ratio

        translation_matrix = np.eye(3, dtype=float)
        translation_matrix[0][2] = -self.width / 2.0
        translation_matrix[1][2] = -self.height / 2.0

        translation_inv_matrix = np.eye(3, dtype=float)
        translation_inv_matrix[0][2] = self.width / 2.0
        translation_inv_matrix[1][2] = self.height / 2.0

        affine_scaled_matrix = translation_inv_matrix @ scaling_matrix @ translation_matrix

        for i in range(self.num_frames):
            success, frame = video.read()
            if not success:
                print(f"unable to read frame {i}/{self.num_frames}")
                exit()
            augmented_b = np.vstack((update_transforms[i], [0, 0, 1]))
            affine_warp = (affine_scaled_matrix @
                           np.linalg.inv(augmented_b))[:2, :]
            smooth_frame = cv2.warpAffine(
                frame, affine_warp, (self.width, self.height))
            out_video.write(smooth_frame)

        out_video.release()

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
            model += 0.9 <= p[i, 2]
            model += p[i, 2] <= 1.1
            model += p[i, 5] >= 0.9
            model += p[i, 5] <= 1.1
            model += -0.1 <= p[i, 3]
            model += p[i, 3] <= 0.1
            model += -0.1 <= p[i, 4]
            model += p[i, 4] <= 0.1
            model += -0.1 <= p[i, 3] + p[i, 4]
            model += p[i, 3] + p[i, 4] <= 0.1
            model += -0.05 <= p[i, 2] - p[i, 5]
            model += p[i, 2] - p[i, 5] <= 0.05

            # Inclusion Constraints
            # (0, 0)T ≤ CR_i * pt ≤ (w, h)T
            for x, y in crop_bounds:
                model += 0 <= p[i, 0] + p[i, 2] * x + p[i, 3] * y
                model += p[i, 0] + p[i, 2] * x + p[i, 3] * y <= self.width
                model += 0 <= p[i, 1] + p[i, 4] * x + p[i, 5] * y
                model += p[i, 1] + p[i, 4] * x + p[i, 5] * y <= self.height

        print("Running solver")
        model.solve()
        smooth_transforms = []
        if model.status == 1:
            print("solution found")
            for i in range(self.num_frames):
                smooth_transforms.append(
                    np.array([[p[i, 2].varValue, p[i, 3].varValue, p[i, 0].varValue],
                              [p[i, 4].varValue, p[i, 5].varValue, p[i, 1].varValue]])
                )
        else:
            print("unable to find solution")
            return None

        return smooth_transforms

    # computes the linear motion model (affine transform homography) between each pair of frames and returns each in a list
    def estimate_per_frame_motion_transforms(self):
        video = cv2.VideoCapture(self.input_file)

        # first transform is just the identity matrix
        transforms = [np.eye(3, 3)[:2, :]]

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
