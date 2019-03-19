import numpy as np


def KDTree(points, depth=11, rnd_split=True, gamma=10.):
    dim = points.shape[1]
    order = []
    split_axis = [[] for _ in range(depth)]

    def split(idxs, crt_depth):
        if crt_depth >= depth:
            order.append(idxs)
            return
        crt_points = points[idxs]
        rng = np.max(crt_points, 0) - np.min(crt_points, 0)
        rng /= np.linalg.norm(rng)
        if rnd_split:
            prob = np.exp(gamma * rng)
            prob /= prob.sum()
        else:
            prob = np.zeros(dim)
            prob[np.argmax(rng)] = 1

        # choose the axis
        ax = np.random.choice(np.arange(dim), None, p=prob)

        idxs_arr = list(idxs)
        idxs_arr.sort(key=lambda idx: points[idx][ax])
        idxs = np.array(idxs_arr)
        split(idxs[:idxs.size // 2], crt_depth + 1)
        split(idxs[idxs.size // 2:], crt_depth + 1)
        split_axis[crt_depth].append(ax)

    split(np.arange(points.shape[0]), 0)
    return order, list(reversed(split_axis))


if __name__ == '__main__':
    points = np.random.rand(8, 3)
    print(points)
    print(KDTree(points, 2, rnd_split=False))
