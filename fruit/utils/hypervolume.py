import numpy as np


# A port from WFG’s hypervolume code
class Point(object):
    def __init__(self):
        self.objs = {}

    def print(self):
        print(self.objs)


class Front(object):
    def __init__(self):
        self.n_points = 0
        self.n_objs = 0
        self.points = {}


class HVCalculator(object):
    """
    Calculates hypervolume, which is used in multi-objective RL.
    """
    def __init__(self):
        self.max_m = 0  # points
        self.max_n = 0  # objectives
        self.n = 0      # objs
        self.fs = []    # memory management
        self.fr = 0     # current depth
        self.fr_max = -1
        self.ref = None

    @staticmethod
    def dominates(point1, point2):
        for i in range(len(point1)):
            if point1[i] < point2[i]:
                return False
        return True

    @staticmethod
    def get_hyper_volume(data):
        return HVCalculator.get_volume_from_array(data, [0.,0.,0.])

    @staticmethod
    def get_volume_from_array(data, ref_point):
        hvc = HVCalculator()
        front = Front()
        front.n_objs = 0

        for points in data:
            if HVCalculator.dominates(points, ref_point):
                p = Point()
                front.n_objs = 0

                for point in points:
                    p.objs[str(front.n_objs)] = point
                    front.n_objs = front.n_objs + 1

                front.points[str(front.n_points)] = p
                front.n_points = front.n_points + 1

        new_ref_point = Point()
        for i in range(len(ref_point)):
            new_ref_point.objs[str(i)] = ref_point[i]
        hvc.set_ref_point(new_ref_point)

        if front.n_points == 0:
            front.points[str(front.n_points)] = new_ref_point
            front.n_points = front.n_points + 1
            front.n_objs = len(new_ref_point.objs)

        return hvc.get_volume(front)

    def set_ref_point(self, point):
        self.ref = point

    def get_volume(self, ps):
        if ps.n_points > self.max_m:
            self.max_m = ps.n_points

        for i in range(ps.n_points):
            if len(ps.points[str(i)].objs) > self.max_n:
                self.max_n = len(ps.points[str(i)].objs)

        if self.ref is None:
            self.ref = Point()
            for i in range(self.max_n):
                self.ref.objs[str(i)] = 0.

        self.n = self.max_n
        return self.hv(ps)

    def hv(self, ps):
        volume = 0.
        for i in range(ps.n_points):
            volume = volume + self.exclhv(ps, i)
        return volume

    def exclhv(self, ps, p):
        volume = self.inclhv(ps.points[str(p)])
        if ps.n_points > p + 1:
            self.make_dominated_bits(ps, p)
            volume = volume - self.hv(self.fs[self.fr-1])
            self.fr = self.fr - 1
        return volume

    def inclhv(self, p):
        volume = 1.
        for i in range(self.n):
            volume = volume * np.abs(p.objs[str(i)] - self.ref.objs[str(i)])
        return volume

    def worse(self, x, y):
        if self.beats(y, x):
            return x
        else:
            return y

    def beats(self, x, y):
        if x > y:
            return True
        else:
            return False

    def make_dominated_bits(self, ps, p):

        if self.fr > self.fr_max:
            self.fr_max = self.fr
            front = Front()
            for i in range(self.max_m):
                point = Point()
                front.points[str(i)] = point
            self.fs.insert(self.fr, front)

        z = ps.n_points - 1 - p   # 1

        for i in range(z):
            for j in range(self.n):
                t = p + 1 + i
                x = ps.points[str(p)].objs[str(j)]
                y = ps.points[str(t)].objs[str(j)]

                self.fs[self.fr].points[str(i)].objs[str(j)] = self.worse(
                    x, y
                )

        self.fs[self.fr].n_points = 1
        for i in range(z):
            if i == 0:
                continue
            j = 0
            keep = True
            while j < self.fs[self.fr].n_points and keep:
                ret = self.dominates_2_way(
                    self.fs[self.fr].points[str(i)],
                    self.fs[self.fr].points[str(j)]
                )
                if ret == -1:
                    t = self.fs[self.fr].points[str(j)]
                    self.fs[self.fr].n_points = self.fs[self.fr].n_points - 1
                    self.fs[self.fr].points[str(j)] = self.fs[self.fr].points[str(self.fs[self.fr].n_points)]
                    self.fs[self.fr].points[str(self.fs[self.fr].n_points)] = t
                    break
                elif ret == 0:
                    j = j + 1
                    break
                else:
                    keep = False

            if keep:
                t = self.fs[self.fr].points[str(self.fs[self.fr].n_points)]
                self.fs[self.fr].points[str(self.fs[self.fr].n_points)] = self.fs[self.fr].points[str(i)]
                self.fs[self.fr].points[str(i)] = t
                self.fs[self.fr].n_points = self.fs[self.fr].n_points + 1

        self.fr = self.fr + 1

    def dominates_2_way(self, p, q):
        for i in range(self.n):
            k = self.n - i - 1
            if self.beats(p.objs[str(k)], q.objs[str(k)]):
                for j in range(k):
                    l = k - j - 1
                    if self.beats(q.objs[str(l)], p.objs[str(l)]):
                        return 0
                return -1
            elif self.beats(q.objs[str(k)], p.objs[str(k)]):
                for j in range(k):
                    l = k - j - 1
                    if self.beats(p.objs[str(l)], q.objs[str(l)]):
                        return 0
                return 1
        return 2


if __name__ == '__main__':
    data = [[1.,1.,4.], [2.,2.,2.], [3., 3., 3.]]
    ref = [0.,0.,0.]
    print(HVCalculator.get_volume_from_array(data, ref))

