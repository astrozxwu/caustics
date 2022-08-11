# compute caustics in coordinate of (m1,-s/2,0) and (m2,s/2,0) where q = m2/m1, s is the separation in thetaE units
import cmath
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

def caustics(s,q,nrot=200,resample=False):
    z1 = -s/2
    z2 = s/2
    m1 = 1/(1+q)
    m2 = 1-m1
    # check type : 1 - wide , 0 resonate , -1 close
    t = 0
    if s > dw(q):
        t = 1
        _key = lambda x:-cmath.phase(x[1])
    elif s < dc(q):
        t =-1
        _key = lambda x:-x[1].imag
    phi_l = np.linspace(0,2*np.pi*(1-1/nrot),nrot)
    caustics = []
    critical = []
    for phi in phi_l: 
        ephi_c = cmath.exp(1.j*phi)
        coeff = [-(m1*z2**2.+m2*z1**2.)*ephi_c+z1**2.*z2**2.,\
                (2.*z2*m1+2.*z1*m2)*ephi_c,\
                2.*z1*z2-ephi_c,\
                0.,\
                1.] 
        solves = np.polynomial.polynomial.polyroots(coeff)
        tcaus = np.array([lenseq(i,z1,z2,m1,m2) for i in solves]) 
        if phi == 0:
            if t == -1:
                solves,tcaus = zip(*sorted(zip(solves,tcaus),key=lambda x:-x[1].imag))
                solves,tcaus = list(solves),list(tcaus)
                if tcaus[1].real > tcaus[2].real:
                    solves[1],solves[2] = solves[2],solves[1]
                    tcaus[1],tcaus[2] = tcaus[2],tcaus[1]
            if t == 0:
                r = sorted(solves,key = lambda x:x.imag)
                _key = lambda x:-cmath.phase(x[1]-center)
                center = (r[1]+r[2])/2
                solves,tcaus = zip(*sorted(zip(solves,tcaus),key=lambda x:x[1].imag))
                solves,tcaus = list(solves),list(tcaus)
                order = [1,3,2,0]
                if tcaus[1].real > tcaus[2].real:
                    order[0],order[2] = 2,1
                solves,tcaus = [solves[i] for i in order],[tcaus[i] for i in order]
            if t == 1:
                solves,tcaus = zip(*sorted(zip(solves,tcaus),key=lambda x:(x[1].real)))
                order = [0,2,3,1]
                solves,tcaus = [solves[i] for i in order],[tcaus[i] for i in order]
        else:
            solves,tcaus = zip(*sorted(zip(solves,tcaus),key=_key))
            solves,tcaus = list(solves),list(tcaus)
        critical.append(solves)
        caustics.append(tcaus)
    cut = 0 
    critical = np.array(critical[cut:]+critical[:cut]).T
    caustics = np.array(caustics[cut:]+caustics[:cut]).T
    if not resample:
        return critical,caustics 
    if t == 0:
        #resonate type
        # input : phi_l , 
        # image list :
        _phi = np.concatenate([phi_l,phi_l+2*np.pi])
        _critic = np.concatenate([critical[0],critical[1]])
        freal = interp1d(_phi, _critic.real, kind='cubic',bounds_error=False,fill_value='extrapolate')
        fimag = interp1d(_phi, _critic.imag, kind='cubic',bounds_error=False,fill_value='extrapolate')
        #plt.scatter(_phi,_critic.real)
        #plt.show()
        _dzdphi = dzdphi(_critic,s,q)
        _dzetadphi = abs(_dzdphi+np.exp(1j*_phi)*_dzdphi.conjugate())
        fdsdphi = interp1d(_phi, _dzetadphi, kind='cubic',bounds_error=False,fill_value='extrapolate')


        __phi = np.linspace(0,4*np.pi,10*nrot)
        __s = cumtrapz(fdsdphi(__phi),__phi,initial=0)
        fphis = interp1d(__s,__phi, kind='cubic',bounds_error=False,fill_value='extrapolate')

        __s_resample = np.linspace(__s[0],__s[-1],nrot)
        __phi_resample = fphis(__s_resample)

        nreal = freal(__phi_resample)
        nimag = fimag(__phi_resample)
        ncritical = np.append((nreal+nimag*1j),np.flip(nreal+nimag*-1j))
        ncaustics = lenseq(ncritical,z1,z2,m1,m2) 
        return [ncritical],[ncaustics]
    else:
        _phi = phi_l
        _critic1,_critic2 = critical[0],critical[1]
        freal1 = interp1d(_phi, _critic1.real, kind='cubic',bounds_error=False,fill_value='extrapolate')
        fimag1 = interp1d(_phi, _critic1.imag, kind='cubic',bounds_error=False,fill_value='extrapolate')
        freal2 = interp1d(_phi, _critic2.real, kind='cubic',bounds_error=False,fill_value='extrapolate')
        fimag2 = interp1d(_phi, _critic2.imag, kind='cubic',bounds_error=False,fill_value='extrapolate')
        _dzdphi1 = dzdphi(_critic1,s,q)
        _dzetadphi1 = abs(_dzdphi1+np.exp(1j*_phi)*_dzdphi1.conjugate())
        fdsdphi1 = interp1d(_phi, _dzetadphi1, kind='cubic',bounds_error=False,fill_value='extrapolate')
        _dzdphi2 = dzdphi(_critic2,s,q)
        _dzetadphi2 = abs(_dzdphi2+np.exp(1j*_phi)*_dzdphi2.conjugate())
        fdsdphi2 = interp1d(_phi, _dzetadphi2, kind='cubic',bounds_error=False,                       fill_value='extrapolate')

        __phi = np.linspace(0,2*np.pi,10*nrot)
        __s1 = cumtrapz(fdsdphi1(__phi),__phi,initial=0)
        __s2 = cumtrapz(fdsdphi2(__phi),__phi,initial=0)
        fphis1 = interp1d(__s1,__phi, kind='cubic',bounds_error=False,fill_value='extrapolate')
        fphis2 = interp1d(__s2,__phi, kind='cubic',bounds_error=False,fill_value='extrapolate')

        __s_resample1 = np.linspace(__s1[0],__s1[-1],nrot)
        __phi_resample1 = fphis1(__s_resample1)
        
        __s_resample2 = np.linspace(__s2[0],__s2[-1],nrot)
        __phi_resample2 = fphis2(__s_resample2)

        nreal1 = freal1(__phi_resample1)
        nimag1 = fimag1(__phi_resample1)
        nreal2 = freal2(__phi_resample2)
        nimag2 = fimag2(__phi_resample2)
        if t == 1:
            ncritical1 = np.concatenate([nreal1+nimag1*1j,np.flip(nreal1+nimag1*-1j)])
            ncritical2 = np.concatenate([nreal2+nimag2*1j,np.flip(nreal2+nimag2*-1j)])
            ncaustics1 = lenseq(ncritical1,z1,z2,m1,m2)
            ncaustics2 = lenseq(ncritical2,z1,z2,m1,m2)
            return [ncritical1,ncritical2],[ncaustics1,ncaustics2]
        else:
            ncritical1p = nreal1+nimag1*1j
            ncritical1n = nreal1+nimag1*-1j
            ncritical2 = np.concatenate([nreal2+nimag2*1j,np.flip(nreal2+nimag2*-1j)])
            ncaustics1p = lenseq(ncritical1p,z1,z2,m1,m2)
            ncaustics1n = lenseq(ncritical1n,z1,z2,m1,m2)
            ncaustics2 = lenseq(ncritical2,z1,z2,m1,m2)
            return [ncritical1p,ncritical1n,ncritical2],[ncaustics1p,ncaustics1n,ncaustics2]

#    dzetadphi = cmath.exp(j*phi) 
    # the caustics sastify the eq : m1/(z-z1)**2 + m2/(z-z2)**2 = exp(-i*phi)
    # however the z is in the image plane 
    # we try to define a paramter s such that d zeta(s)/d s is an constant
    # we have a parameterization of the caustics with zeta(phi) and we try to find the relation between phi and s

def dzdphi(z,s,q):
    z1 = complex(-s/2,0)
    z2 = complex(s/2,0)
    m1 = 1/(1+q)
    m2 = 1-m1
    return 1j/2*(m2*(z-z1)**2+m1*(z-z2)**2)/(m2*(z-z1)**3+m1*(z-z2)**3)*(z-z1)*(z-z2)


def lenseq(z,z1,z2,m1,m2):
    zeta_c = z.conjugate()+m1/(z1-z)+m2/(z2-z)
    return zeta_c.conjugate()

def dc(q):
    # dc^8 = (1+q)^2/27/q*(1-dc^4)^3
    c = (1+q)**2/27/q
    dc4 = np.polynomial.polynomial.polyroots([-c,3*c,1-3*c,c])
    for i in dc4:
        if np.isreal(i):
            return i.real**0.25
    return 'no solve found , pls check input'
def dw(q):
    return ((1+q**(1/3))**3/(1+q))**0.5


if __name__ == '__main__':
    critic,caustic = caustics(0.89,0.029822227,nrot=1000,resample=1)
    for c in caustic:
#        plt.scatter(c.real[0],c.imag[0])
        plt.plot(c.real,c.imag)#,s=1)
    plt.show()
