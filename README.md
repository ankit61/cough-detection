# cough-detection

This is a repository to detect the act of "coughing" using deep learning. The input to the model is a video, and the output is the probability that there was "coughing" in the video. As shown below, a probability-time graph can also be generated for better understanding.

## Instructions to run

Run the following commands to set up the environment for running:

```bash
#clone and set up
git clone https://github.com/ankit61/cough-detection.git
cd cough-detection/
mkdir models/

#install dependencies
sudo add-apt-repository -y ppa:savoury1/ffmpeg4
sudo add-apt-repository -y ppa:savoury1/graphics
sudo add-apt-repository -y ppa:savoury1/multimedia
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install 
sudo apt-get install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavutil-dev libswscale-dev libswresample-dev
pip3 install -r requirements.txt
sudo add-apt-repository --remove ppa:savoury1/ffmpeg4
sudo add-apt-repository --remove ppa:savoury1/graphics
sudo add-apt-repository --remove ppa:savoury1/multimedia
```

## Executing code

Sample results can be generated in the following manner:
```bash
python3 main.py --mode gen_result --data-dir /content/cough-detection/data/test --conv3d-load-path models/MultiStreamDNN00_checkpoint_20_9030.pth --conv3d-load-path models/MultiStreamDNN00new_checkpoint_19.pth
```
You must download the two models named above from the Google Drive link below. Also, you must follow the file naming conventions inside ```YOUR_DATA_DIR```. You must first use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body landmarks for your videos. Then, the video with the body landmarks must be suffixed with ```_v.mp4``` and the original video must be suffixed by ```_a.mp4```. For example, if you want to run the code on ```YOUR_VIDEO.mp4```, then you must generate body landmarks and save them in ```YOUR_VIDEO_v.mp4``` and rename the original video ```YOUR_VIDEO_a.mp4```. The visual features would be extracted from the ```*_v.mp4``` videos and the audio features will be extracted from ```*_a.mp4```.

Running the above command would produce a ```YOUR_VIDEO_v.json```, ```YOUR_VIDEO_v_label.png```, ```YOUR_VIDEO_v_prob.png``` for every video in ```YOUR_DATA_DIR```. ```YOUR_VIDEO_v_label.png``` is a predicted label vs time graph and ```YOUR_VIDEO_v_prob.png``` is a probability vs time graph.

If you want to train or test the model, you can use the following command:
```bash
#for training
python3 main.py --mode train --data-dir /content/cough-detection/data/test --conv3d-load-path models/MultiStreamDNN00_checkpoint_20_9030.pth --conv3d-load-path models/MultiStreamDNN00new_checkpoint_19.pth

#for testing
python3 main.py --mode test --data-dir /content/cough-detection/data/test --conv3d-load-path models/MultiStreamDNN00_checkpoint_20_9030.pth --conv3d-load-path models/MultiStreamDNN00new_checkpoint_19.pth
```

Note again that ```YOUR_DATA_DIR``` must have videos named in the same format as described above of ```*_v.mp4``` and ```*_a.mp4```. In addition, ```YOUR_DATA_DIR``` must also have a ```labels.json```. The format of ```labels.json``` file is the following:

```json
{
"file1_v.mp4": [1, 2, 3],
"file2_v.mp4": [1],
"file3_v.mp4": []
}
```

This ```labels.json``` indicates that there was coughing in ```file1_v.mp4``` between 1 - 2, 2 - 3, 3 - 4 seconds. Similarly, there was coughing in ```file2_v.mp4``` between 1 - 2 seconds and there was no coughing in ```file3_v.mp4```.


## Trained Models
Accessible at https://drive.google.com/open?id=1s_K1Ju2haDDMzyxUejwhasOY6BnKuL08

## YouTube Videos

Sample videos playlist: 
 - Submission 1: https://www.youtube.com/playlist?list=PLzTd-IE8CXpX3N-rEbUMjfIlHXNDH-90S
 - Submission 2: https://www.youtube.com/playlist?list=PLzTd-IE8CXpWKQMGInhXw3QRCfXQwJsNl
 - Submission 3: https://www.youtube.com/playlist?list=PLzTd-IE8CXpW4FmoskUIUOcJOIKDRlZy6
 - Submission 4: https://www.youtube.com/playlist?list=PLzTd-IE8CXpXAKg7AAiar0wi_bOHz5uNU

Video explaining how to run: https://youtu.be/Ov9urimfum8

