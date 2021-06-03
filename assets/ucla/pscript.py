import os
from PIL import Image

zz = len([x for x in os.listdir('.') if x.endswith('jpg')])
for enn, i in enumerate([x for x in os.listdir('.') if x.endswith('jpg')]):
    img = Image.open(i)
    img = img.crop((0, 0, 1024, 1024))
    fn = f'./{enn}-crop.jpg'
    print(fn)
    img.save(fn)