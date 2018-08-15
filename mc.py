import numpy as np

from kinematics import *

def gen_phi():
    return 2.*np.pi*np.random.rand()

def gen_costh():
    return -1.+2.*np.random.rand()

def gen_two_body_decay_products_rest_frame(M, m1, m2):
    """
    Randomly selects four-vectors in the rest frame of the decay particle 
    with two body kinemtics. The angular distribution is flat for two body decays.
    M = mass of decaying particle; m1, m2 = daughter particle masses;
    Output is two four-vectors p1, p2 corresponding to m1, m2
    """
    if m1+m2>M:
        return -1,-1
    
    costh = gen_costh()
    if np.random.rand() > 0.:
        sinth = np.sqrt(1.-costh**2.)
    else:
        sinth = -np.sqrt(1.-costh**2.)

    ph = gen_phi()
    
    p = np.sqrt((M**2 - (m1+m2)**2)*(M**2 - (m1-m2)**2))/(2.*M)
    
    E1 = np.sqrt(m1**2 + p**2)
    E2 = np.sqrt(m2**2 + p**2)
    
    v1 = np.array([E1, p*sinth*np.cos(ph), p*sinth*np.sin(ph), p*costh])
    v2 = np.array([E2, -p*sinth*np.cos(ph), -p*sinth*np.sin(ph), -p*costh])
    
    return v1, v2



def get_rotation_matrix(v):
    """
    Find a rotation matrix s.t. R v = |v|(0,0,1)
    """
    
    vx = v[0]; vy = v[1]; vz = v[2]
    
    # Find first rotation to set y component to 0
    if np.fabs(vx) > 0. and np.fabs(vy) > 0.:
        ta = np.fabs(vy/vx)
        a = np.arctan(ta)
        
        if vx > 0. and vy > 0.:
            a = -a
        if vx < 0. and vy > 0.:
            a = -(np.pi - a)
        if vx < 0. and vy < 0.:
            a = -(np.pi + a)
        if vx > 0. and vy < 0.:
            a = -(2.*np.pi - a)
              
        ca = np.cos(a)
        sa = np.sin(a)
    elif np.fabs(vy) > 0.:
        ca = 0.
        sa = 1.
    else:
        ca = 1.
        sa = 0.
        
    Ra = np.array([[ca, -sa, 0.],[sa,ca,0.],[0.,0.,1.]])

    #sa = 0.
    #ca = 1.
    
    # Find second rotation to set x component to 0
    vxp = vx*ca - vy*sa
    if np.fabs(vz) > 0. and np.fabs(vxp) > 0.:
        tb = np.fabs(vxp/vz)
        b = np.arctan(tb)

        if vz > 0. and vxp > 0.:
            b = -b
        if vz < 0. and vxp > 0.:
            b = -(np.pi - b)
        if vz < 0. and vxp < 0.:
            b = -(np.pi + b)
        if vz > 0. and vxp > 0.:
            b = -(2.*np.pi - b)

        cb = np.cos(b)
        sb = np.sin(b)
    elif vxp > 0.:
        cb = 0.
        sb = -1.
    elif vxp < 0.:
        cb = 0.
        sb = 1.
    else:
        cb = 1.
        sb = 0.
        
    Rb = np.array([[cb, 0., sb],[0., 1., 0.],[-sb, 0., cb]])
    
    #return Rb
    
    #print "Rb v = ", np.matmul(Rb,v)
    
    return np.matmul(Rb,Ra)

def boost_back(p_mother, p_daughter):
    """
    Assuming p_daughter is given in the rest frame of p_mother, 
    boost p_daughter back (along z only!)
    """
    p_reverse = np.array([p_mother[0],-p_mother[1],-p_mother[2],-p_mother[3]])
    return boost(p_reverse, p_daughter)

def gen_two_body_decay_products(p_mother, M, m1, m2):
    """
    Generate proper decay product four_vectors in the 
    frame where the mother particle has four momentum p_mother
    """
    three_mom = p_mother[1:4]
    
    R = get_rotation_matrix(three_mom)
    three_mom_rotated = np.matmul(R,three_mom)

    p_mother_rotated = np.array([p_mother[0],three_mom_rotated[0],three_mom_rotated[1],three_mom_rotated[2]])
    #print p_mother_rotated
    # Get four vectors in mother rest frame
    p1, p2 = gen_two_body_decay_products_rest_frame(M, m1, m2)
    
    # Boost back 
    p1 = boost_back(p_mother_rotated, p1)
    p2 = boost_back(p_mother_rotated, p2)
    
    # Rotate back
    Rinv = R.transpose()
    #print Rinv
    
    p1_three_vec = np.matmul(Rinv,p1[1:4])
    p2_three_vec = np.matmul(Rinv,p2[1:4])
    
    p1 = np.array([p1[0],p1_three_vec[0],p1_three_vec[1],p1_three_vec[2]])
    p2 = np.array([p2[0],p2_three_vec[0],p2_three_vec[1],p2_three_vec[2]])
    
    return p1, p2

def gen_three_body_decay_products_rest_frame(M, m1, m2, m3):
    """
    Randomly selects four-vectors in the rest frame of the decay particle 
    with three body kinemtics. The matrix element and phase space weight are not yet applied. 
    M = mass of decaying particle; m1, m2, m3 = daughter particle masses;
    Output are three four-vectors p1, p2, p3 corresponding to m1, m2, m3
    """
    if m1+m2+m3 > M:
        return -1,-1,-1
    
    # Phase space decomposed into product of two body phase spaces
    # First set corresponds to a decay M -> m1 + sqrt(s23)
    s23_max = (M - m1)**2
    s23_min = (m2 + m3)**2
    s23 = s23_min + (s23_max - s23_min) * np.random.rand()
    m23 = np.sqrt(s23)
    p1, p23 = gen_two_body_decay_products_rest_frame(M, m1, m23)
    
    # Second set corresponds to decay sqrt(s23) -> m2 + m3
    # Generate momenta in the M rest frame 
    p2, p3 = gen_two_body_decay_products(p23, m23, m2, m3)
    
    return p1, p2, p3

def beta(m1, m2, s12):
    x = m1**2/s12
    y = m2**2/s12
    
    return np.sqrt(1. - 2.*(x+y) + (x-y)**2)

def get_three_body_weight(M, m1, m2, m3, p1, p2, p3, mesq):
    p23 = p2 + p3
    s23 = lor_prod(p23,p23)
    m23 = np.sqrt(s23)
    
    beta1 = beta(m1, m23, M**2)
    beta23 = beta(m2, m3, s23)
    
    me_weight = mesq(M, m1, m2, m3, p1, p2, p3)

    # This is the volume of the phase space integral with beta factors factored out
    # ds23 dcosth23
    V =  2.* M**2 * (1.-(m1+m2+m3)/M)*(1.-(m1-m2-m3)/M)/(4.*np.pi*8.*np.pi*8.*np.pi)
    
    return V*beta1*beta23*me_weight

def get_average_weight(weights):
    """
    Average weight
    """
    w = np.array(weights)
    return np.sum(w)/len(w)

def get_sample_variance(weights):
    """
    Variance of the sample
    """
    w = np.array(weights)
    w2 = np.power(w,2.)
    w_avg = get_average_weight(weights)
    
    return np.sum(w2)/len(w2) - w_avg**2

def get_average_variance(weights):
    """
    Variance of the mean
    """
    return get_sample_variance(weights)/len(weights)

def gen_three_body_decay_products(p_mother, M, m1, m2, m3):
    """
    Generate proper decay product four_vectors in the 
    frame where the mother particle has four momentum p_mother
    """
    three_mom = p_mother[1:4]
    
    R = get_rotation_matrix(three_mom)
    three_mom_rotated = np.matmul(R,three_mom)

    p_mother_rotated = np.array([p_mother[0],three_mom_rotated[0],three_mom_rotated[1],three_mom_rotated[2]])

    # Get four vectors in mother rest frame
    p1, p2, p3 = gen_three_body_decay_products_rest_frame(M, m1, m2, m3)
    
    # Boost back 
    p1 = boost_back(p_mother_rotated, p1)
    p2 = boost_back(p_mother_rotated, p2)
    p3 = boost_back(p_mother_rotated, p3)
    
    # Rotate back
    Rinv = R.transpose()
    
    p1_three_vec = np.matmul(Rinv,p1[1:4])
    p2_three_vec = np.matmul(Rinv,p2[1:4])
    p3_three_vec = np.matmul(Rinv,p3[1:4])
    
    p1 = np.array([p1[0],p1_three_vec[0],p1_three_vec[1],p1_three_vec[2]])
    p2 = np.array([p2[0],p2_three_vec[0],p2_three_vec[1],p2_three_vec[2]])
    p3 = np.array([p3[0],p3_three_vec[0],p3_three_vec[1],p3_three_vec[2]])
    
    return p1, p2, p3

