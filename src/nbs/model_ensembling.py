import sys
sys.path.append('../utils/')

import libraries
from libraries import *
import utils_functions
from utils_functions import *

# checking
print("Number of detected GPUs: ", len(get_available_gpus()))
print("Memory check: ", psutil.virtual_memory())
print("Keras version: ", keras.__version__)


parser = argparse.ArgumentParser()

parser.add_argument('--models', type=str, default=None)
parser.add_argument('--metrics', type=str, default=None)
# parser.add_argument('--test_data_folder', type=bool, default=False)
parser.add_argument('--from_dir', type=bool, default=False)
params = parser.parse_args()
print(params)
