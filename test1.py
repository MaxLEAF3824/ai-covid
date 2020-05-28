import os

dir = "D:\\Coding\\Others\\COVID"
NonCOVID_file_num = len(os.listdir(os.path.join(dir, "train_NonCOVID")))
print(NonCOVID_file_num)
