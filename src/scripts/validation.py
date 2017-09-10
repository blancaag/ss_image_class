import sys
sys.path.append('../utils/')

import libraries
from libraries import *

import utils
from utils import *

reload(libraries)
from libraries import *

# checking
print(get_available_gpus())
print(psutil.virtual_memory())
print(keras.__version__)

print("test...")
