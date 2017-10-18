# BestMoments
BestMoments is a Python script that uses OpenCV to search for the best moments on a video file.
It generates snapshots for sharing.

## First Features:
- Smile detection: script will search for faces and smiles and will rank by number of smiles found on each frame.
- Frame sharpness: you can define a minimum frame sharpness or quality, bellow that limit the script will ignore the frame.

## Basic Usage
```sh
python bestmoments.py -i sample.mp4 -ms 50
```

## Command line arguments
<pre>usage:
    bestmoments.py -i sample.mp4

options:
    -h, --help                      show this help message and exit
    -i FILE, --input-file FILE      path video file
    -ms 50, --min-sharpness 50      minimum frame sharpness to consider
</pre>


## Sample Results
![Alt text](screenshot.png?raw=true)

Sample video from: https://videos.pexels.com/videos/fun-at-a-fair-491
