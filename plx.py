import numpy as np

def plx_config(t0,alpha,delta):
    global ecc,peri,north,east,xpos,ypos,qe2,qn2,qe1,qn1,qe0,qn0,t0par
    t0par = t0
    ecc = 0.0167
    vernal = 2719.55
    offset = 75
    radian = 180/np.pi
    p = 2*np.pi
    peri = vernal - offset # Perihelion time icrs frame
    spring = np.array([1.,0.,0.])
    summer = np.array([0.,0.9174,0.3971])
    north = np.array([0,0,1])
    star = np.array([np.cos(alpha/radian)*np.cos(delta/radian),
            np.sin(alpha/radian)*np.cos(delta/radian),
            np.sin(delta/radian)])
    east = np.cross(north,star)
    east = east/np.sum(east**2)**0.5 
    north = np.cross(star,east)

    phi = (1-offset/365.25)*p
    psi = getpsi(phi)
    cos = (np.cos(psi)-ecc)/(1-ecc*np.cos(psi))
    sin = -(1-cos**2)**0.5
    xpos = spring*cos+summer*sin
    ypos = -spring*sin+summer*cos

    phi = (t0+1-peri)/365.25*p
    psi = getpsi(phi)
    sun = xpos*(np.cos(psi)-ecc)+ypos*(np.sin(psi))*(1-ecc**2)**0.5
    qn2 = np.dot(sun,north)
    qe2 = np.dot(sun,east)

    phi = (t0-1-peri)/365.25*p
    psi = getpsi(phi)
    sun = xpos*(np.cos(psi)-ecc)+ypos*(np.sin(psi))*(1-ecc**2)**0.5
    qn1 = np.dot(sun,north)
    qe1 = np.dot(sun,east)

    phi = (t0-peri)/365.25*p
    psi = getpsi(phi)
    sun = xpos*(np.cos(psi)-ecc)+ypos*(np.sin(psi))*(1-ecc**2)**0.5
    qn0 = np.dot(sun,north)
    qe0 = np.dot(sun,east)

def geta(t):
    phi = (t-peri)/365.25*2*np.pi
    psi = getpsi(phi)
    sun = xpos*(np.cos(psi)-ecc)+ypos*(np.sin(psi))*(1-ecc**2)**0.5
    qn = np.dot(sun,north)
    qe = np.dot(sun,east)

    qn -= qn0 + (qn2-qn1)*(t-t0par)/2
    qe -= qe0 + (qe2-qe1)*(t-t0par)/2

    return qn,qe

def getpsi(phi):
    psi = phi
    for i in range(4):
        fun = psi-ecc*np.sin(psi)
        dif = phi-fun
        psi += dif/(1-ecc*np.cos(psi))
    return psi

if __name__ == '__main__':
    plx_config(1,1,1)
    print(np.array([geta(i) for i in range(5)]).T)
