from pathlib import Path
import random

src = Path("../synthetic_data/ECG_jump")
train = src / "train"
test = src / "test"
train.mkdir(exist_ok=True)
test.mkdir(exist_ok=True)

files = list(src.glob("*.npz"))
random.shuffle(files)

split = int(0.8 * len(files))
for f in files[:split]:
    f.rename(train / f.name)
for f in files[split:]:
    f.rename(test / f.name)
