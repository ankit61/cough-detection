import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

NUM_CLASSES                 = 1

PRINT_FREQ                  = 20
INTERMITTENT_OUTPUT_FREQ    = 5 # Num batches between outputs

#HYPERPARAMETERS
BATCH_SIZE                  = 32
EPOCHS                      = 300
LR                          = 0.05
WEIGHT_DECAY                = 0

#DATASET
CHUNK_SIZE                  = 1
AUDIO_SAMPLE_RATE           = 48000
RESAMPLED_AUDIO_SAMPLE_RATE = 16000
INPUT_FRAME_WIDTH           = 224
VIDEO_FPS                   = 30
MEAN                        = [0.6226, 0.6329, 0.6279]
STD                         = [0.22181073012818833, 0.22090722034374521, 0.21610182784974308]

MIN_LEARNING_RATE           = 0.000001

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = '/media/sf_data'
