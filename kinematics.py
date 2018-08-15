import numpy as np

def get_mom_from_lhe(rec):
    e, px, py, pz = rec['e'], rec['px'], rec['py'], rec['pz']
    return np.array([e,px,py,pz])

def pt(p):
    px = p[1]
    py = p[2]
    return np.sqrt(px**2 + py**2)

def rap(p, maxrap=10.):
    px, py, pz = p[1], p[2], p[3]
    
    if px == 0. and py == 0.:
        return maxrap
    if pz == 0.:
        return 0.
    
    pt = np.sqrt(px**2 + py**2)
    
    th = np.arctan(pt/pz)
    if th < 0.:
        th = th + np.pi
    
    return -np.log(np.tan(th/2.))

def theta(p):
    px, py, pz = p[1], p[2], p[3]
    pt = np.sqrt(px**2 + py**2)
    th = np.arctan(pt/pz)
    if th < 0.:
        th = th + np.pi
    return th

def lor_prod(p,v):
    return p[0]*v[0] - p[1]*v[1] - p[2]*v[2] - p[3]*v[3]

def boost(p,v):
    rsq = np.sqrt(lor_prod(p,p))
    v0 = lor_prod(p,v)/rsq
    c1 = (v[0] + v0)/(rsq + p[0])
    boosted_v = [v0, v[1] - c1*p[1], v[2] - c1*p[2], v[3] - c1*p[3]]
    
    return np.array(boosted_v)
