import numpy as np
from easydict import EasyDict as edict

root = edict()
cfg = root


root.gpu_ids = '0,1,5,1,0,3,1,3,7,6,1,0,1,0'
root.device = None
root.timestamp = None
root.root_outputdir = 'output'
root.param_weights_dir_MIR = 'output/weights/MIR/'
root.outputdir =None
root.avalanche_dir = False



root.seed = 25
