import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

NUM_CLASSES                 = 1

PRINT_FREQ                  = 3
INTERMITTENT_OUTPUT_FREQ    = 5 # Num batches between outputs
SAVE_FREQ                   = 5

#HYPERPARAMETERS
BATCH_SIZE                  = 24
EPOCHS                      = 300
LR                          = 0.005
WEIGHT_DECAY                = 4e-4
MOMENTUM                    = 0.9
DROPOUT_PROB                = 0.5
FRAME_3D_CONVS              = 10
HOLISTIC_FEATURES_LEN       = 1024
N_MFCCS                     = 100
N_MELS                      = 300

#ENSEMBLE HYPERPARAMS
ENSEMBLE_LRS                = [LR, 0.0001]
ENSEMBLE_MOMENTUMS          = [MOMENTUM, MOMENTUM]
ENSEMBLE_WEIGHT_DECAYS      = [WEIGHT_DECAY, 12e-4]

#DATASET
CHUNK_SIZE                  = 1
AUDIO_SAMPLE_RATE           = 48000
RESAMPLED_AUDIO_SAMPLE_RATE = 16000
INPUT_FRAME_WIDTH           = 224
VIDEO_FPS                   = 30
MEAN                        = [0.0059, 0.0051, 0.0042]
STD                         = [0.0474, 0.0456, 0.0358]
VISUAL_SUFFIX               = '_v.mp4'
AUDIO_SUFFIX                = '_a.mp4'

MIN_LEARNING_RATE           = 0.000001

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), 'data')
