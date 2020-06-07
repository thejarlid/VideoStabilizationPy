# Video Stabilization

This is an implementation of the methods presented in [this research paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37041.pdf) from Google which aims to stabilise video using linear programming and L1 optimization of the optimal camera path. I also wrote up a blog post on this paper and specifically my experience implementing this [here](https://thejarlid.github.io/posts/VideoStabilization.html). 

# Results

![smooth](/results/stable_0_150.gif)

![motion](/results/motion_300_600.png)

## Setup

This repo uses python 3's virtual environment and pip to allow a sandbox environment of its own dependencies which have been frozen into the requirements.txt file. 

Inside the cloned directory run the following in the terminal:

```bash
python3 -m venv venv/               # you can change venv/ to whatever you want your virtual environment directory to be called
source venv/bin/activate            # to start the virtual environment 
pip install -r requirements.txt     # get the dependencies 
```