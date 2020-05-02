#!/bin/bash

#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.com/ankit61/cough-detection"
REPO="cough-detection"

OPENPOSE_VIDEO="YOUR_VIDEO_NAME_v.mp4" # name must end with "_v.mp4" (refer GitHub README
AUDIO_STREAM_VIDEO="YOUR_VIDEO_NAME_a.mp4" # name must end with "_a.mp4" (refer GitHub README)

TEMP_DIR="temp_825006585/"

OUT_JSON=echo "${OPENPOSE_VIDEO##*/}" | cut -d. -f1
OUT_JSON="${OUT_JSON}.json"

OUT_PNG=echo "${OPENPOSE_VIDEO##*/}" | cut -d. -f1
OUT_PNG="${OUT_JSON}.png"

UIN_JSON="825006585.json"
UIN_PNG="825006585.png"

git clone $GIT_REPO_URL
cd $REPO

# Install dependencies
sudo add-apt-repository -y ppa:savoury1/ffmpeg4
sudo add-apt-repository -y ppa:savoury1/graphics
sudo add-apt-repository -y ppa:savoury1/multimedia
sudo apt-get -q update
sudo apt-get install -q ffmpeg
sudo apt-get install -q libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavutil-dev libswscale-dev libswresample-dev
pip3 install -r requirements.txt
sudo add-apt-repository --remove -y ppa:savoury1/ffmpeg4
sudo add-apt-repository --remove -y ppa:savoury1/graphics
sudo add-apt-repository --remove -y ppa:savoury1/multimedia


mkdir $TEMP_DIR # make temp directory to store data
cp $OPENPOSE_VIDEO $TEMP_DIR
cp $OPENPOSE_VIDEO $TEMP_DIR

echo $OPENPOSE_VIDEO

python3 main.py --data-dir $TEMP_DIR --mode gen_result --model_type conv3D_MFCCs --conv3d-load-path models/MultiStreamDNN00_checkpoint_20_9030.pth

rm -r $TEMP_DIR

cp OUT_JSON $UIN_JSON
cp OUT_PNG $UIN_PNG
