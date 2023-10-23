import shutil

a = ["a.tif", "b.tif"]
b = ["a.txt", "b.txt"]
c = []

for img in a:
    for label in b:
        if label[0] == img[0]:
            c.append(label)
print(c)
