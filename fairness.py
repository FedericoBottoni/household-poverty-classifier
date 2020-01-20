import sys
sys.path.append('configs')
sys.path.append('src')
sys.path.append('src\\shared')
sys.path.append('src\\utils')
from fairness_check import check_flist

if len(sys.argv) <= 1:
    print('input path is missing')
else:
    check_flist(sys.argv[1])