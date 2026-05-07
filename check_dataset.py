print("\nChecking dataset...\n")

import os

for root, dirs, files in os.walk("dataset"):
    print(root, "->", len(files), "images")