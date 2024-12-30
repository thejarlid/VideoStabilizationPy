# Video Stabilization

This is an implementation of the methods presented in [this research paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37041.pdf) from Google which aims to stabilise video using linear programming and L1 optimization of the optimal camera path.

Beginning originally as a computer vision class final project in 2018 I had the idea to take frame by frame homographies, plot the x and y jitter, and then take a gausian filter over some window of frames to produce a smooth path. This had flaws in stabilization as it got rid of high frequency jitter that spanned a small window of frames but didn't produce inherently smooth videos. I later discovered this paper which fits constant, linear, and parabolic paths to the overall motion.

This implementation was done as a self study on the research paper to better understand the methods proposed

# Results

On the left is the original video and on the right is my smoothed clip. 

![smooth](/results/results.gif)

![motion](/results/path.png)


## Setup

```bash
python3 -m venv venv/               # you can change venv/ to whatever you want your virtual environment directory to be called
source venv/bin/activate            # to start the virtual environment 
pip install -r requirements.txt     # get the dependencies 
```

## Usage

```bash
python3 src/video_stabilization.py input.mp4 -o output.mp4 -c 0.9

usage: L1 Video Stabilizer [-h] [-o OUT_FILE] [-c CROP_RATIO] [-p] in_file

positional arguments:
  in_file               input video file to stabilize

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_FILE, --out_file OUT_FILE
                        name of the output file default will be the input file name with the suffix _stable
  -c CROP_RATIO, --crop_ratio CROP_RATIO
                        crop ratio for the crop window [0, 1]
  -p, --plot            flag for whether to save and output the plot graphs
```

# Credits
Matlab implementation which influenced some of the decisions in this implementation https://github.com/ishit/L1Stabilizer/tree/master
