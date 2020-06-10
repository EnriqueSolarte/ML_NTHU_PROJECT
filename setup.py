import sys
import os

print("         ML COURSE FINAL PROJECT         ")
print("                Fall 2020                ")

main_project = os.path.dirname(__file__)
sys.path.append(os.path.dirname(__file__))
print("\n# PYTHON PATH:")
[print(path) for path in sys.path]

print("\n# GLOBAL VARIABLES:")

DATA_DIR = os.path.join(main_project, "data")
DATA_TRAIN_DIR = os.path.join(main_project, "data", "train_images")
DATA_TEST_DIR = os.path.join(main_project, "data", "test_images")
SRC_DIR = os.path.join(main_project, "src")

print("DATA_DIR: {}".format(DATA_DIR))
print("DATA_TRAIN_DIR: {}".format(DATA_TRAIN_DIR))
print("DATA_TEST_DIR: {}".format(DATA_TEST_DIR))
print("SRC_DIR: {}".format(SRC_DIR))
