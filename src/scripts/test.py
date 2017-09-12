import sys
sys.path.append('../utils/')

import libraries
from libraries import *

import utils
from utils import *


# checking
print(get_available_gpus())
print(psutil.virtual_memory())
print(keras.__version__)

print(test)
