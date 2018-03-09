from __future__ import division, print_function

class ValueHeuristic(object):
    def __init__(self, rmax, rmin, discount_rate,
                 vmin=None, vmax=None):
        self.rmax = rmax
        if rmin is None:
            rmin = -rmax
        self.rmin = rmin
        self.discount_rate = discount_rate
        if vmax is None:
            self._vmax = rmax / (1 - discount_rate)
        else:
            self._vmax = vmax

        if vmin is None:
            self._vmin = rmin / (1 - discount_rate)
        else:
            self._vmin = vmin

    def vmax(self, s=None, d=None):
        return self._vmax

    def vmin(self, s=None, d=None):
        return self._vmin

    def set_fsss(self, fsss):
        self.fsss = fsss

    def __hash__(self):
        return hash((
            self.rmax,
            self.rmin,
            self.discount_rate
        ))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False