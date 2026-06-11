import io
import struct
import numpy as np
import MDAnalysis as mda


# XTC:
#    What                 Bytes Format
#  0. Magic number (4)    0 -  4 l
#  1. Atoms (4)           4 -  8 l
#  2. Step  (4)           8 - 12 l 
#  3. Time (4)           12 - 16 f
#  4. Box (9*4)          16 - 52 fffffffff
# 13. Atoms (4)          52 - 56 l           (checked to be equal to b.)
# 14. Precision (4)      56 - 60 f
# 15. Extent (6*4)       60 - 84 llllll      (MIN: x,y,z, MAX: x,y,z)
# 21. smallidx (4)       84 - 88 l
# 22. Size in bytes (4)  88 - 92 l 
# 23. Coordinates (compressed)
# unpacked   = unpack(">lllfffffffffflfllllllll", header)


class XTCIndexer(io.FileIO):
    def __init__(self, filename, top=None, selection=None):
        super().__init__(filename, 'rb')
        if top is not None:
            self.top = mda.Universe(top)
            self.atoms = self.top.atoms
        else:
            self.top = None
            self.atoms = None
        self.headers = []
        self.tag = self.read(8)
        self.size = self.seek(0, 2)
        self.n_atoms = struct.unpack('>l', self.tag[4:])[0]
        self.positions = [self.seek(88) - 88]
        self.pos = 0
        while self.positions[-1] < self.size:
            self.nextframe()

    def __len__(self):
        return len(self.positions) - 1

    def nextframe(self):
        fsize = struct.unpack('>l', self.read(4))[0]
        fsize += -fsize % 4
        self.positions.append(self.seek(fsize + 88, 1) - 88)

    def read(self, frames=None, selection=None):
        if frames is None:
            frames = range(len(self.positions) - 1)
        
        for frame in frames:
            self.seek(self.positions[frame])
            yield self._read_frame()

    def _read_frame(self):
        """Read a single frame of data."""
        header = self.read(92)  # Read the header
        coords = self.read(self.n_atoms * 3 * 4)  # Read the coordinates
        coords = np.frombuffer(coords, dtype=np.float32).reshape((self.n_atoms, 3))
        coords = np.round(coords, 3)  # Round to three decimal places
        return coords