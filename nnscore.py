import sys
sys.path.append('configs')
sys.path.append('src')
sys.path.append('src\\shared')
from train import score

if len(sys.argv) <= 1:
    print('input path is missing')
else:
    score(sys.argv[1])