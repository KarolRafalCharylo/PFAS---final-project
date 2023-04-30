from tqdm import tqdm
import time

for i in tqdm(range(5), position=0, desc="i", leave=False, colour='green', ncols=80):
    for j in tqdm(range(10), position=1, desc="j", leave=False, colour='red', ncols=80):
        time.sleep(0.5)