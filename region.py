class Region:
    def __init__(self, l, u, r, d):
        self.l = float(l)
        self.u = float(u)
        self.r = float(max(l, r))
        self.d = float(max(u, d))
        self.probability = 0.0
        self.idx = 0

    def unwrap(self):
        return self.l, self.u, self.r, self.d

    def intersect(self, other):
        return Region(max(self.l, other.l),
                      max(self.u, other.u),
                      min(self.r, other.r),
                      min(self.d, other.d))

    def area(self):
        return (self.r - self.l) * (self.d - self.u)
