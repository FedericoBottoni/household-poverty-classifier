import sys
sys.path.append('configs')
sys.path.append('src')
sys.path.append('src\\shared')
sys.path.append('src\\utils')
from auto_ml import sample_hps

if len(sys.argv) <= 1:
    print('input path is missing')
else:
    sample_hps(sys.argv[1])