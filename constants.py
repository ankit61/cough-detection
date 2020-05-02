import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

NUM_CLASSES                 = 1

PRINT_FREQ                  = 3
INTERMITTENT_OUTPUT_FREQ    = 5  # Num batches between outputs
SAVE_FREQ                   = 5

# HYPERPARAMETERS
BATCH_SIZE                  = 32
EPOCHS                      = 300
DROPOUT_PROB                = 0.5
FRAME_3D_CONVS              = 10
HOLISTIC_FEATURES_LEN       = 1024
N_MFCCS                     = 40
N_MELS                      = 300

LRS                         = [0.0001, 0.005, 0.005]
MOMENTUMS                   = [0.9, 0.9, 0.9]
WEIGHT_DECAYS               = [4e-4, 4e-4, 4e-4]

# DATASET
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

assert N_MELS == 300, 'MEAN, STD COMPUTED FOR N_MELS=300'
AUDIO_MEAN                  = [3.24896550178527832031]
AUDIO_STD                   = [42.03905868530273437500]

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), 'models')

if not os.path.exists(MODELS_BASE_DIR):
    os.mkdir(MODELS_BASE_DIR)

DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))), 'data')
