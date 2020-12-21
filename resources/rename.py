import glob
import os

filenames = glob.glob("*.png")
for i, filename in enumerate(filenames):
    new_name = str(i)
    os.rename(filename, f"{new_name}.png")