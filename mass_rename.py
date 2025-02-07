import os

directory = r"D:\SleepApneaPtData\set_1_complete"
# Assumes directory only contains images
num = 0
for dirpath, dirnames, filenames in os.walk(directory):
    for file in filenames:
        filepath = directory + "\\" + file
        output_path = directory + "\\" + str(num).zfill(4) + ".bmp"
        os.rename(filepath, output_path)
        num += 1