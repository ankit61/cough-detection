import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

NUM_CLASSES                 = 2

PRINT_FREQ                  = 20
INTERMITTENT_OUTPUT_FREQ    = 5 # Num batches between outputs

#HYPERPARAMETERS
BATCH_SIZE                  = 64
EPOCHS                      = 300
LR                          = 0.05
WEIGHT_DECAY                = 0

#DATASET
CHUNK_SIZE                  = 1
AUDIO_SAMPLE_RATE           = 48000
RESAMPLED_AUDIO_SAMPLE_RATE = 16000
INPUT_FRAME_WIDTH           = 100
VIDEO_FPS                   = 30

MIN_LEARNING_RATE           = 0.000001

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = '/media/sf_data'
