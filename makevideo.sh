ffmpeg -r 30 -start_number 1 -i './data/output/frame%d.jpg' -c:v libx264 -vf "fps=30,format=yuv420p, scale=640:-2" final.mp4
