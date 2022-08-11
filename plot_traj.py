import numpy as np
import matplotlib.pyplot as plt
import caustics as cs
from plx import *

# load param
with open('param','r') as u:
    exec('param = '+u.read())
for i in param.keys():
    exec(f'{i} = {param[i]}')
alpha = 2*np.pi - alpha


RA,Dec = 159.67677,-61.26380
plx = True 
npoints = 2000
# geometry configuration
offset_x = s*(-0.5+1/(1+q)) if s < 1 else s/2-q/(1+q)/s
offset_y = 0

# traj coord
t = np.linspace(t0-1.5*tE,t0+1.5*tE,npoints)
tau = (t-t0)/tE
dtau,du0 = 0,0
if plx:
    plx_config(t0,RA,Dec)
    qn,qe = np.array([geta(i) for i in t]).T
    dtau = pi1*qn+pi2*qe
    du0 = -pi1*qe+pi2*qn
taup = tau + dtau
u0p = u0 - du0

xs = taup*np.cos(alpha)+u0p*np.sin(alpha) #- offset_x
ys =-taup*np.sin(alpha)+u0p*np.cos(alpha) #- offset_y

critical,caustics = cs.caustics(s,q,resample=1,nrot=1000)

# plot configuration 

fig, ax = plt.subplots(figsize=(6,8))
for i in caustics:
    real = i.real + offset_x
    imag = i.imag + offset_y
    ax.plot(real,imag,color='b')
ax.plot(xs,ys,'k--')
ax.arrow((xs[int(npoints*2/3)+1]+xs[int(npoints*2/3)])/2, (ys[int(npoints*2/3)+1]+ys[int(npoints*2/3)])/2, \
       xs[int(npoints*2/3)+1]-xs[int(npoints*2/3)], ys[int(npoints*2/3)+1]-ys[int(npoints*2/3)],\
       color='black',lw=0, length_includes_head=True, head_width=.05)
#ax.grid(True)
ax.axis('equal')

# append some special points
t = [9630,9670,9700,t0]
for i in t:
    tau = (i-t0)/tE
    qn,qe = geta(i)
    dtau = pi1*qn+pi2*qe
    du0 = -pi1*qe+pi2*qn
    taup = tau + dtau
    u0p = u0 - du0
    xs = taup*np.cos(alpha)+u0p*np.sin(alpha) #- offset_x
    ys =-taup*np.sin(alpha)+u0p*np.cos(alpha) #- offset_y
    circle = plt.Circle((xs,ys),rho/2,color='r',fill=False)
    ax.add_patch(circle)
    ax.annotate('{:.1f}'.format(i), xy=[xs,ys], xytext=(-5, -15),
   textcoords='offset points', size='small',)
plt.tight_layout()
plt.show()
