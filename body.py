class Body(object):
    m = None
    v = []
    r = []
    def _init_(self):
        pass
    def info(self):
        print("m: {}".format(self.m))
        print("vO =  [{0},{1},{2}]".format(self.v[0],self.v[1],self.v[2]))
        print("rO =  [{0},{1},{2}]".format(self.r[0],self.r[1],self.r[2]))
