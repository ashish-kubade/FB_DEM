import struct
import sys
import numpy as np
def dem2ply(H, plyName, cellSize):	
            
    w = H.shape[1]
    h = H.shape[0]
    
    fply = open(plyName+'.ply', 'wb')
    fply.write(bytes('ply\n', 'ascii'))
    fply.write(bytes('format binary_little_endian 1.0\n', 'ascii'))
    fply.write(bytes('element vertex %d\n' % (w*h), 'ascii'))
    fply.write(bytes('property float x\n', 'ascii'))
    fply.write(bytes('property float y\n', 'ascii'))
    fply.write(bytes('property float z\n', 'ascii'))
    fply.write(bytes('element face %d\n' % ((w-1)*(h-1)*2), 'ascii'))
    fply.write(bytes('property list uint8 int32 vertex_index\n', 'ascii'))
    fply.write(bytes('end_header\n', 'ascii'))       
    for i in range(w):
        for j in range(h):  
            fply.write(struct.pack('<fff', cellSize*i, cellSize*j, H[j][i]))            
    for i in range(w-1):
        for j in range(h-1):                 
            fply.write(struct.pack('<Biii', 3, i*h + j, (i+1)*h + j, i*h + j + 1))
            fply.write(struct.pack('<Biii', 3, (i+1)*h + j, (i+1)*h + j + 1, i*h + j + 1))
    fply.close()   

if __name__=='__main__':
    dem_path = sys.argv[1]
    dem = np.load(dem_path)
    ply_name = dem_path[:-4]
    dem2ply(H=dem, plyName = ply_name, cellSize = 2)