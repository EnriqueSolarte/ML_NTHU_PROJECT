import sys
import os

print("ML COURSE FINAL PROJECT")
print("NTHU Fall 2020")

main_project = os.path.dirname(os.path.dirname(__file__))
sys.path.append(main_project)

print("\n# PYTHON PATH:")
[print(path) for path in sys.path]

print("\n# GLOBAL VARIABLES:")

# ! This line is for local server
# SERVER_DATA = os.path.join(main_project, "data")
# ! This line is for remote server
SERVER_DATA = '/media/NFS/kike/ml-course/data'
DATA_DIR = os.path.join(SERVER_DATA)
DATA_TRAIN_DIR = os.path.join(SERVER_DATA, "train_images")
DATA_VALIDATION_DIR = os.path.join(SERVER_DATA, "validation_images")
DATA_TEST_DIR = os.path.join(SERVER_DATA, "test_images")
DIR_SRC = os.path.join(main_project, "src")
DIR_LOG = os.path.join(main_project, "log")
print("DATA_DIR: {}".format(DATA_DIR))
print("DATA_TRAIN_DIR: {}".format(DATA_TRAIN_DIR))
print("DATA_TEST_DIR: {}".format(DATA_TEST_DIR))
print("DIR_SRC: {}".format(DIR_SRC))
print("DIR_LOG {}".format(DIR_LOG))
sys.path.append(DIR_SRC)
CLASSES = ['normal', 'void', 'horizontal_defect', 'vertical_defect', 'edge_defect', 'particle']
