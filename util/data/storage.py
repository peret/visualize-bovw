import numpy as np
import struct
import zlib
import sys

#unsigned char, e.g. for SIFT descriptor matrix
CV_8U = 0
CV_32S = 4
CV_32F = 5

class Keypoint:

    def __init__(self, x=0.0, y=0.0, size=1.0, angle=-1.0, octave=0, response=0.0, class_id=-1):
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle
        self.octave = octave
        self.response = response
        self.class_id = class_id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)

def keypoints_from_file(fname):
    with open(fname, "rb") as f:
        num, = struct.unpack("i", f.read(4))
        size = 28 # a keypoint has seven 4-byte members
        result = [Keypoint( *(struct.unpack("ffffifi", f.read(size))) ) for _ in range(num)]
        return result

def keypoints_to_file(keypoints, fname):
    with open(fname, "wb") as f:
        f.write(struct.pack("i", len(keypoints)))
        for k in keypoints:
            f.write(struct.pack("ffffifi", k.x, k.y, k.size, k.angle, k.octave, k.response, k.class_id))

def load_matrix(fname):
    with open(fname, "rb") as f:
        compressed, = struct.unpack("?", f.read(1))
        datasize, = struct.unpack("L", f.read(8))

        if compressed:
            compdatasize, = struct.unpack("L", f.read(8))
            data = zlib.decompress(f.read(compdatasize),)
            rows, cols, mtype = struct.unpack("iii", data[:12])

            if mtype == CV_8U:
                struct_param = "B"
                size = 1
                dtype = np.uint8
            elif mtype == CV_32S:
                struct_param = "I"
                size = 4
                dtype = np.uint32
            elif mtype == CV_32F:
                struct_param = "f"
                size = 4
                dtype = np.float32
            else:
                raise NotImplementedError("This matrix data type is not supported.")

            matrix = np.zeros((rows, cols), dtype)

            bytes_read = 12
            row = 0

            while bytes_read < len(data):
                col, = struct.unpack("i", data[bytes_read:bytes_read + 4])
                bytes_read += 4
                if col == -1:
                    row += 1
                else:
                    matrix[row,col] = struct.unpack(struct_param, data[bytes_read:bytes_read + size])[0]
                    bytes_read += size
            return matrix

        else:
            raise NotImplementedError("The uncompressed matrix format is currently not supported.")

def save_matrix(mat, fname):
    with open(fname, "wb") as f:
        f.write(struct.pack("?", True))
        data = ""

        if mat.dtype == np.uint8:
            struct_param = "B"
            mtype = CV_8U
        elif mat.dtype == np.uint32:
            struct_param = "I"
            mtype = CV_32S
        elif mat.dtype == np.float32:
            struct_param = "f"
            mtype = CV_32F
        else:
            raise NotImplementedError("Unsupported matrix dtype")

        rows, cols = mat.shape
        data += struct.pack("iii", rows, cols, mtype)

        for r in range(rows):
            for c in range(cols):
                if mat[r,c] != 0.0:
                    data += struct.pack("i", c)
                    data += struct.pack(struct_param, mat[r,c])
            data += struct.pack("i", -1)

        # not sure this is the correct bytesize
        f.write(struct.pack("L", len(data)))
        comp = zlib.compress(data, 9)
        f.write(struct.pack("L", len(comp)))
        f.write(comp)