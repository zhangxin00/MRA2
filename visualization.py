# -*- coding: UTF-8 -*-
import numpy
from PIL import Image
import binascii
import os
from PIL import Image
import numpy
def getMatrixfrom_bin(filename,width):
    with open(filename, 'rb') as f:
        content = f.read()
        # print(content)
    hexst = binascii.hexlify(content)

    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])

    rn = len(fh)/width
    rn=int(rn)

    fh = numpy.reshape(fh[:rn*width],(-1,width))
    fh = numpy.uint8(fh)
    return fh,content

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.exe':
                L.append(os.path.join(root, file))
    return L

url='dataset'
print(len(file_name(url)))
for i in file_name(url):
    name=i.split('\\')[1]
    print(name)
    array,h=getMatrixfrom_bin(i, 512)
    print(array)
    print(array.shape)

    im = Image.fromarray(array)
    im.save('xxxx')
