from tensorpack.dataflow import *
import h5py
import numpy as np
from KDTree import KDTree

def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, False)
    return rot


class MyDataflow(RNGDataFlow):
    def __init__(self, data_path, rnd_rot, augment):
        t_files = []
        with open(data_path, 'r') as flist:
            for line in flist:
                t_files.append(line.rstrip())

        f = []
        for file in t_files:
            f.append(h5py.File(file))

        data = f[0]['data'][:]
        label = f[0]['label'][:]

        for i in range(1, len(f)):
            data = np.concatenate((data, f[i]['data'][:]), axis=0)
            label = np.concatenate((label, f[i]['label'][:]), axis=0)

        for ff in f:
            ff.close()

        print(data.shape, label.shape)
        self.points = data
        self.label = label
        self.rnd_rot = rnd_rot
        self.augment = augment

    def __len__(self):
        return self.points.shape[0]

    def __iter__(self):
        while True:
            idx = self.rng.randint(len(self))
            points = self.points[idx]
            label = self.label[idx]
            if self.augment:
                rot = rnd_rot()
                points = np.einsum('ij,nj->ni', rot, points)
                points += np.random.rand(3)[None, :] * 0.05
                points = np.einsum('ij,nj->ni', rot.T, points)

            rand_rot = rnd_rot() if self.rnd_rot else np.eye(3)
            points = points @ rand_rot
            order, split_axis = KDTree(points, 11)
            yield (points, np.asarray(order), np.asarray(label[0]), *split_axis)

