import MDAnalysis as mda
import numpy as np
from matplotlib import tri

import time

import struct
import io
import os


class Timer(list):
    def __call__(self, msg=None, nopar=False):
        
        # With a message, run as context
        if msg is not None:
            self.append([msg, time.time()])
            return self
        
        # Without a message, run as decorator
        def inner(func):
            run = [0]
            def wrapper(*args, **kwargs):
                run.append(run[-1] + 1)
                msg = f'{func.__name__}[{run[-1]}]'
                msg += ' ...' if nopar else f'{args} {kwargs}'
                with self(msg=msg):
                    func(*args, **kwargs)
            return wrapper
        return inner
        
    def __repr__(self):
        out = ''.join(f'{lbl}: {e-s:.6f}s\n' for lbl, s, e in self)
        self.clear()
        return out
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self[-1].append(time.time())        


class XTCIndexer(io.FileIO):
    def __init__(self, filename):
        super().__init__(filename, 'rb')
        self.tag = self.read(8)
        self.size = self.seek(0, 2)
        self.n_atoms = struct.unpack('>l', self.tag[4:])[0]
        self.positions = [ self.seek(0) ]
        while self.positions[-1] < self.size:
            self.nextframe()
    
    def nextframe(self):
        self.seek(88, 1)
        fsize = struct.unpack('>l', self.read(4))[0]
        fsize += -fsize % 4
        self.positions.append(self.seek(fsize, 1))
        
    def write(self, filename, start=0, stop=None, step=None):
        if isinstance(start, int):
            start = range(len(self.positions))[start:stop:step]
        with open(filename, 'wb') as out:
            for f in start:
                self.seek(self.positions[f])
                out.write(self.read(self.positions[f+1] - self.positions[f]))
        np.savez(
            f'.{filename}_offsets.npz', 
            offsets=self.positions[:-1],
            size=self.positions[-1],
            ctime=os.path.getctime(filename),
            n_atoms=self.n_atoms
        )


def dim2pbc(arr: np.ndarray) -> np.ndarray:
    '''
    Convert unit cell definition from PDB CRYST1 format to lattice definition.
    '''
    
    lengths = arr[:, :3]
    angles = arr[:, 3:] * (np.pi / 180)
    
    cosa = np.cos(angles)
    sing = np.sin(angles[:, 2])    
    
    pbc = np.zeros((len(arr), 9))
    pbc[:, 0] = lengths[:, 0]
    pbc[:, 3] = lengths[:, 1] * cosa[:, 2]
    pbc[:, 4] = lengths[:, 1] * sing    
    pbc[:, 6] = lengths[:, 2] * cosa[:, 1]
    pbc[:, 7] = lengths[:, 2] * (cosa[:, 0] - cosa[:, 1] * cosa[:, 2]) / sing
    pbc[:, 8] = (lengths[:, 2] ** 2 - (pbc[:, 6:8] ** 2).sum(axis=1)) ** 0.5
    
    return pbc.reshape((-1, 3, 3))
    

class TrajectoryWithPBC(np.ndarray):
    '''
    A simple class to contain a trajectory with the coordinates and 
    unit cell definitions for all frames read in as numpy arrays.
    '''
    attributes = (
        'universe', # The mda.Universe from the selected atoms
        'atoms', # The atoms from the universe
        'times', # The times of the frames
        'pbc', # The PBC lattice matrices for all frames
        'centers', # The centers 
        'orientations', # The orientations
        'rgyr_' # Radii of gyration (set by align)
        'rmsd_' # Root mean square deviation (set by align)
    )
    
    def __new__(cls, tpr: str, trj: str, selection=None, start=None, stop=None, step=None):

        # Bookkeeping: MDA stuff
        if selection is None:
            selection = 'all'
        try:
            atomgroup = mda.Universe(tpr, trj).select_atoms(selection)
        except OSError:
            raise OSError(f'XDR read error in {tpr}:{trj}')

        # Content: times, pbc, coordinates
        pbc, times, coords = [], [], []
        for frame in atomgroup.universe.trajectory[start:stop:step]:
            pbc.append(atomgroup.dimensions.copy())
            coords.append(atomgroup.positions.copy())
            times.append(frame.time)

        # The object
        obj = np.array(coords).view(cls)
        obj.pbc = dim2pbc(np.array(pbc))
        obj.times = np.array(times)
        obj.universe = mda.Merge(atomgroup)
        obj.atoms = obj.universe.atoms
        obj.centers = np.zeros((len(coords), 3))
        obj.orientations = np.outer(np.ones(len(coords)), np.eye(3)).reshape((-1, 3, 3))
                
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        for attr in self.attributes:
            setattr(self, attr, getattr(obj, attr, None))
            
    def __matmul__(self, other):
        trj = (self.view(np.ndarray) @ other).view(self.__class__)
        self._transfer_attributes(trj)
        trj.pbc = self.pbc @ other
        return trj
        
    def __getitem__(self, item):
        if item is None:
            # Override the behaviour of adding an axis
            # to allow NoneType selections
            return self
        if isinstance(item, (int, slice)):
            result = super().__getitem__(item)
            self._transfer_attributes(result)
            result.pbc = self.pbc[item]
            return result
        if isinstance(item, str):
            item = self.universe.select_atoms(item)
        if isinstance(item, mda.AtomGroup):
            result = self[:, item.ix]
            result.universe = mda.Merge(result.universe.atoms[item.ix])
            result.atoms = result.universe.atoms
            return result
        return super().__getitem__(item)
        
    def __and__(self, selection):
        '''Return the trajectory for the intersection of the atomgroup and selection'''
        return self[:, self.universe.select_atoms(selection).ix]
            
    def _transfer_attributes(self, other):
        for attr in self.attributes:
            setattr(other, attr, getattr(self, attr, None))
        
    @property
    def coords(self):
        return self.view(np.ndarray)
    
    @property
    def bcoords(self):
        # Integers indicate cells, fractions positions in cells
        result = self @ np.linalg.inv(self.pbc)
        result.pbc = self.pbc
        return result
    
    @property
    def angles(self):
        result = self @ (2 * np.pi * np.linalg.inv(self.pbc))
        result.pbc = self.pbc
        return result
    
    @property
    def skewed(self):
        result = self @ self.pbc
        result.pbc = self.pbc
        return result

    @property
    def angskewed(self):
        result = self @ (0.5 / np.pi * self.pbc)
        result.pbc = self.pbc
        return result
    
    def align(self, selection=None, reference=None, plane=None):
        '''Align the trajectory with respect to reference and selection'''
        # PBC safe centering
        self.origin(selection)

        fit = self[selection].coords
        natoms = len(fit[0])
        if reference is None:
            reference = fit[0]

        # Radii of gyration (for RMSD)
        rgyr2 = (fit ** 2).sum(axis=(1, 2)) / natoms
        refrg2 = (reference ** 2).sum() / natoms
            
        # Actual fitting
        U, L, V = np.linalg.svd(reference.T @ fit)
        R = U @ V
        result = self @ R.transpose((0, 2, 1))
        
        # Bookkeeping
        result.orientations = R
        rmsd2 = rgyr2 + refrg2 - 2 * L.sum(axis=1) / natoms
        rmsd2[rmsd2 < 0] = 0
        result.rmsd_ = rmsd2 ** 0.5
        result.rgyr_ = rgyr2 ** 0.5
        
        return result
        
    def alignxy(self, selection=None, reference=None):
        '''Align the trajectory with respect to reference and selection'''
        # PBC safe centering
        self.origin(selection)
        
        fit = self[selection].coords[:, :, :2]
        natoms = len(fit[0]) 
        if reference is None:
            reference = fit[0]
        
        # Actual fitting
        U, L, V = np.linalg.svd(reference[:, :2].T @ fit)
        R = np.zeros_like(self.pbc)
        R[:, :2, :2] = U @ V
        R[:, 2, 2] = 1
        result = self @ R.transpose((0, 2, 1))
        
        # Bookkeeping
        result.orientations = R
        return result   
        
    def origin(self, selection=None):
        '''Center the selection at the origin for all frames (PBC safe)'''
        if selection is None:
            selection = 'all'
        # Use box coordinate angles to unambiguously define center of mass
        angles = self[selection].angles
        cosa, sina = np.cos(angles), np.sin(angles)
        centers = np.arctan2(sina.mean(axis=1), cosa.mean(axis=1))[:, None] @ (0.5 / np.pi * self.pbc)
        self.centers = centers
        self -= centers
        return self

    def center(self, selection=None):
        '''Center the selection in the center of the triclinic cell (PBC safe'''
        return self.origin(selection) + 0.5 * self.pbc.sum(axis=1)[:, None]

    def inbox(self):
        '''Put all particles in triclinic unit cell'''
        boxed = self @ np.linalg.inv(self.pbc)
        boxed = (boxed - np.floor(boxed)) @ self.pbc
        boxed.pbc = self.pbc
        return boxed
    
    def originbox(self):
        '''Put all particles in triclinic unit cell around origin'''
        boxed = self @ np.linalg.inv(self.pbc)
        boxed = (boxed - np.floor(boxed + 0.5)) @ self.pbc
        boxed.pbc = self.pbc
        return boxed
    
    def compact(self, around=None):
        if around is not None:
            self.origin(around)
        B = self.bcoords.reshape((-1, 3))
        # Shift origin to middle of box
        B += 0.5
        B -= np.floor(B)
        x, y = B[:, :2].coords.T
        check = (y < 0.5 - x) | (y > 1.5 - x)
        up = y > x
        halfx = 0.5 * x
        twox = 2 * x
        B[check &  up & (y > 1.25 - halfx), 1] -= 1
        B[check & ~up & (y < 0.25 - halfx), 1] += 1
        B[check &  up & (y < 0.5 - twox), 0] += 1
        B[check & ~up & (y > 2.5 - twox), 0] -= 1
        # Shift middle of hexagon to origin and transform
        B = ((B.reshape(self.shape) - 0.5) @ self.pbc).view(self.__class__)
        self._transfer_attributes(B)
        return B
        
    def molbox(self, selection):
        # This may be expensive...
        # Voxelized will be faster
        ...
        
    def split(self, what):
        '''Split the trajectory according to mda.atomgroup.split'''
        return [ self[ag] for ag in self.universe.atoms.split(what) ]
        

class PBCHexagonalGrid(np.ndarray):
    '''
    A near hexagonal grid with PBC in two dimensions
    '''
    attributes = (
        'pbc', # The PBC lattice matrices for all frames
        'avg', # The mean lattice
        'unit', # The unit cells for all frames
        'nbox', # The lattice in units (equal for all frames)
        'z', # The height
    )

    def __new__(cls, pbc, resolution=1):
        # Determine the hixel dimensions and number
        unit = resolution * np.array(((1, 0), (0.5, 0.75 ** 0.5)))
        mean = pbc.mean(axis=0)[:2, :2]
        nbox = (np.linalg.inv(unit) @ mean).astype(int)
        # This cell is the closest to hexagonal that fits the lattice
        # for the specified resolution
        unit = np.linalg.inv(nbox) @ pbc[:, :2, :2]
        # This is the grid that fits the lattice
        grid = np.mgrid[:nbox[0, 0], :nbox[1, 1]].T.reshape((-1, 2)).view(cls)
        grid.pbc = pbc
        grid.avg = mean
        grid.nbox = nbox
        grid.unit = unit
        grid.z = np.zeros_like(grid[0])
        return grid
        
    def __array_finalize__(self, obj):
        if obj is None:
            return None
        for attr in self.attributes:
            setattr(self, attr, getattr(obj, attr, None))
        
    @property
    def points(self):
        return self @ (np.linalg.inv(self.nbox) @ self.avg)
        
    def bin(self, trj):
        '''Set z values to ...'''
        ...
        
    def kde(self, trj, prop=None, bw=0.01):
        '''Set z values to distance weighted sum of property'''
        # The grid in box coordinates
        G = self @ np.linalg.inv(self.nbox)
        out = np.zeros((len(trj), len(G)))
        # Distance vectors in box coordinates
        for idx, frame in enumerate(trj.bcoords.coords):
            frame = frame[:, None, :2] - G[None, :]
            frame = (frame - np.floor(frame + 0.5)) @ G.avg
            # Weights per atom (frameatoms, gridpoints)
            W = np.exp(-(frame ** 2).sum(axis=2) / bw)
            if prop is None:
                out[idx] = W.sum(axis=0)
            elif isinstance(prop, np.ndarray) and len(prop.shape) == 2:
                out[idx] = (W * prop[idx]).sum(axis=0) / W.sum(axis=0)
            else:
                out[idx] = (W * prop).sum(axis=0) / W.sum(axis=0)
        return out


class HexagonalBins:
    def __init__(self, A: np.ndarray, resolution: float=1, padding: float=0, tol: float=1e-8):

        # Store the data for determining properties - mind the shape
        self.points = A

        # Binning is done over the last axis
        # Tolerance is to ensure everything is _in_ a bin
        inshape = A.shape
        A = A.reshape((-1, 3))
        self.min = A.min(axis=0) - padding - tol
        self.max = A.max(axis=0) + padding + tol
        self.range = self.max - self.min
        self.resolution = resolution
        
        # Bin the data in a way that each bin consists of 2 rectangular
        # subcells that together contain the data from one actual cell: 
        # one central hexagon and the corners of surrounding ones.
        #
        #  ^  +----01---+  
        #  |  |    |    |  
        #  |  |    |    |  
        #  |  |\   |   /|
        #  a  | 00 | 10 |
        #  |  |   \|/   |
        #  |  |    |    |
        #  |  |    |    |
        #  V  00---+----10
        #
        #     <---r/2--->  a = r sqrt(3/4)
        #
        scale = (resolution / 2, resolution * 0.75 ** 0.5)
        F = (A[:, :2] - self.min[:2]) / scale
        # May need a view as base ndarray type
        B = F.view(np.ndarray).astype(int)
        F -= B # Fraxels
        # Make the division line related to y = x
        F[:, 1] *= 3 
        # Reverse the downward lines (checkerboard selection)
        even = (B % 2).sum(axis=1) == 0
        F[even, 0] *= -1
        F[even, 0] += 1
        # Shift the contents in upper halves (y > x + 1) up in register
        B[F[:, 1] > F[:, 0] + 1, 1] += 1
        # Register a shift in even rows for x
        B[0::2, 0] += 1
        B[:, 0] //= 2
         
        # Groups per bin. Empty bins are not taken into account.
        # The counts array is corrected to be complete with
        # zero counts for empty bins, and can be used as mask
        # to set bin-based properties.
        nx, ny = B.max(axis=0) + 1
        N = np.array([ B[:, 0] + B[:, 1] * nx, np.arange(len(B)) ]).T
        N = N[np.argsort(N[:, 0])]
        uniq, idx, count = np.unique(N[:, 0], return_index=True, return_counts=True)
        groups = np.split(N[:, 1], idx[1:])
        
        # This is the corresponding grid (I think)
        G = np.mgrid[:nx, :ny].astype(float)
        G[0, :, 1::2] += 0.5
        G = G.T.reshape((-1, 2))
        G *= (resolution, resolution * 0.75 ** 0.5)
        G += self.min[:2]
                
        # Register attributes. The triangulation is done using 
        # Delauney triangulation. Using the regularity of the 
        # grid may be faster and easily allows excluding
        # empty bins.
        self.counts = np.zeros(nx*ny)
        self.counts[uniq] = count
        self.groups = groups
        self.grid = G 
        self.tri = tri.Triangulation(G[:, 0], G[:, 1])

    def __call__(self, fun, *args, **kwargs):
        prop = np.zeros_like(self.counts)
        points = self.points.reshape((-1, 3))
        prop[self.counts.astype(bool)] = [
            fun(points[g], *args, **kwargs)
            for g in self.groups
        ]
        return prop

    def kde(self, features, resolution=None, **kwargs):
        if resolution is None:
            grid = self.grid
        else:
            grid = hexgrid(self.grid, resolution)
        kwargs['chunkgrid'] = kwargs.get('chunkgrid', True)
        totalweight, featuresums, featureweights = kde(
            grid, self.grid, features, **kwargs
        )
        return grid, totalweight, featuresums, featureweights
    
        
def hexgrid(points, resolution=1.0, padding=0.1):
    # Bounding box, taking into account padding and the shifting of rows
    mnx, mny = points.min(axis=0) - padding - (0.25 * resolution, 0)
    mxx, mxy = points.max(axis=0) + padding + (0.25 * resolution, 0)
    # This gives a hexagonal grid in a very simple way
    G = np.mgrid[mnx:mxx:resolution, mny:mxy:resolution*0.75**0.5]
    # Shift the odd rows
    G[0, :, 1::2] += 0.5*resolution
    # Reshape to an array of 2D points
    G = G.T.reshape((-1, 2))
    return G


def gaussian(A, sd=100):
    return np.exp(-A / sd)


def kde(grid, points, features=None, chunksize=1e7, chunkgrid=False, kernel=gaussian, **kwargs):
    '''Estimate kernel density of points on grid positions'''
    
    # The features should be an array with the same first dimension as the points
    # I.e., there's a value for each point
    if features is not None:
        features = features.reshape((len(features), -1)).T
        featmask = features.astype(bool)
        featweight = np.zeros((len(features), len(grid)))
        featnorm = np.zeros_like(featweight)
    
    if chunkgrid:
        # Determine how many rows to read in a batch so as
        # not to exceed the maximum chunk size
        chunk = max(1, int(chunksize / len(points)))

        weights = np.zeros(len(grid))
        for i in range(0, len(grid), chunk):
            slc = slice(i, i+chunk)
            P = grid[slc]
            # The weight of each point (selected) at each grid point
            W = kernel(((P[:, None] - points[None, :])**2).sum(axis=2))
            weights[slc] = W.sum(axis=1)
            if features is None:
                continue
            for fid, feat in enumerate(features):
                fm = featmask[fid]
                FW = W[:, fm]
                featnorm[fid, slc] = FW.sum(axis=1)
                featweight[fid, slc] += (FW * feat[fm]).sum(axis=1)
    else:
        # Determine how many rows to read in a batch so as
        # not to exceed the maximum chunk size
        chunk = max(1, int(chunksize / len(grid)))

        weights = np.zeros(len(grid))
        for i in range(0, len(points), chunk):
            slc = slice(i, i+chunk)
            P = points[slc]
            # The weight of each point (selected) at each grid point
            W = kernel(((grid[:, None] - P[None, :])**2).sum(axis=2))
            weights += W.sum(axis=1)
            if features is None:
                continue
            for fid, feat in enumerate(features[:, slc]):
                fm = featmask[fid, slc]
                FW = W[:, fm]
                featnorm[fid] += FW.sum(axis=1)
                featweight[fid] += (FW * feat[fm]).sum(axis=1)

    if features is None:
        return weights, None, None

    featnorm[featnorm == 0.0] = 1
    return weights, (featweight / featnorm).T, featnorm.T

