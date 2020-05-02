#!/bin/bash

#Replace the variables with your github repo url, repo name, test video name, json named by your UIN
GIT_REPO_URL="https://github.com/ankit61/cough-detection"
REPO="cough-detection"

BASE_VIDEO_NAME="1082C4FE-F10F-4621-B50E-21C38C4D" # make sure this is in the current working directory

OPENPOSE_VIDEO="${BASE_VIDEO_NAME}_v.mp4" # refer GitHub README
AUDIO_STREAM_VIDEO="${BASE_VIDEO_NAME}_a.mp4" # name must end with "_a.mp4" (refer GitHub README)

TEMP_DIR="temp_825006585/"

OUT_JSON="${BASE_VIDEO_NAME}_v.json"

OUT_PNG="${BASE_VIDEO_NAME}_v_label.png"

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
cp $AUDIO_STREAM_VIDEO $TEMP_DIR

echo $BASE_VIDEO_NAME

python3 main.py --data-dir $TEMP_DIR --mode gen_result --model-type conv3D_MFCCs --conv3d-load-path /content/MultiStreamDNN00_checkpoint_20_9030.pth

rm -r $TEMP_DIR

cp $OUT_JSON $UIN_JSON
cp $OUT_PNG $UIN_PNG
