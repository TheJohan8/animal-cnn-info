import os
from collections import Counter

base_dir = 'Animals'
all_counts = {
    cls: len(os.listdir(os.path.join(base_dir, cls)))
    for cls in os.listdir(base_dir)
}
print(all_counts)
