import mcml_photon_scattering as mcml
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import scipy



'''
---------------------------------------------Monte Carlo Simulation---------------------------------------------
Shoot a number of [numPhotons] photons in multi-layered medium and track photon trajectories as a 3-dimensional
matrix. Properties of multilayered medium are defined in mcml_photon_scattering.py. Photons are launched from a
square beam spot size of 15-by-15um and goes through an aperture of radius [r_aperture]. Three light sources can
be simulated by specifying [srcShape]: 0-Collimated; 1-Lambertian; 2-Gaussian. 3D photon flux output is output
as [md.Flux_xyz].
'''

# Simulation setup
modelName = 'LEDSTACK'     # MCML model name
numPhotons = 10000000        # number of photons used for Monte-Carlo
srcShape = 3    # 0-Collimated; 1-Lambertian; 2-Gaussian; 3-Measured
gausshalfangledeg = 8
cdfResolution = 1000
LEDx = 0.0011  #cm
LEDy = 0.0006  #cm
LEDarea = LEDx*LEDy  # cm2
# r_aperture = 0.00075  # cm
r_aperture = 0.01  # cm  #roughly large aperture that doesn't affect results but allows faster sim
A_aperture = r_aperture**2*np.pi


# Run sim
Tstart = time.time()
md = mcml.MCMLModel(modelName)
md.do_one_run(numPhotons, srcShape, LEDx, LEDy, r_aperture, cdfResolution, gausshalfangledeg)
Tend = (time.time()-Tstart)
print('--------------Elapsed time: %.2f minutes---------------' %(Tend/60))






# Plot results
dz = md.dz
nz = md.nz
dr = md.dr
nr = md.nr
ixy_cent = md.ixy_cent


# 2D cross section
fig = plt.figure(3); plt.ion()  
im = plt.imshow(md.Flux_xyz[ixy_cent]/numPhotons*LEDarea/dr**2, \
    extent=[0,dz*nz*1e4, -dr*nr/2*1e4,dr*nr/2*1e4], norm=colors.LogNorm())
# im = plt.imshow(md.Flux_xyz[md.ixy_cent], norm=colors.LogNorm())
plt.colorbar(im)
plt.clim(1e-3,1)
plt.xlabel('um'); plt.ylabel('um')
plt.show()
figName = 'MeasBeam_6by11um_nph1e7_glass2retina_noAperture_2D'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# 1D cross section
fig = plt.figure(4); plt.ion()
Z0 = np.arange(0,dz*nz,dz)*1e4
Zirr = md.Flux_xyz[ixy_cent][ixy_cent]/numPhotons*LEDarea/dr**2
plt.semilogy(Z0, Zirr)
plt.ylim(3e-4,2)
plt.show()
figName = 'MeasBeam_6by11um_nph1e7_glass2retina_noAperture_1D'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# Beam 2D XZ cross section with contour lines
power = 0.005 # mW
irradiance0 = power/LEDarea  # mW/cm2
extent = [0,dz*nz*1e4, -dr*nr/2*1e4,dr*nr/2*1e4]
levels = np.arange(-30,10,1)

Z = md.Flux_xyz[ixy_cent]/numPhotons*LEDarea/dr**2 * irradiance0
Z = np.maximum(Z,1e-12)
Zlog = np.log10(Z)
Z1d = md.Flux_xyz[ixy_cent][ixy_cent]/numPhotons*LEDarea/dr**2 * irradiance0
Z0 = np.arange(0,dz*nz,dz)*1e4

fig,ax = plt.subplots(); plt.ion()  # 2D cross section
im = ax.imshow(Z, extent=extent, norm=colors.LogNorm())
im.set_clim(1e4, 1)
CS = ax.contour(Zlog, levels, colors='w', extent=extent, linewidths=0.5, linestyles='solid')
cbar = plt.colorbar(im,ax=ax)
cbar.ax.set_ylabel('LED Irradiance $\mathrm{(mW/cm^{2})}$')
plt.xlabel('Axial (um)')
plt.ylabel('Lateral (um)')
plt.ylim([-96,96])
for v in np.logspace(0,3,4):
    i = min(range(len(Z1d)), key=lambda i: abs(Z1d[i]-v))
    # plt.text(Z0[i],Z0[i]*.2,'{0:.0f}'.format(v),color='w')
    plt.text(Z0[i]*0.9,Z0[i]*.5,'{0:.0e}'.format(v),color='w')
plt.show()
figName = 'ledPwr5uW_MeasBeam_15by19um_nph1e7_glass2retina_noAperture_2D_contour'
plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


np.sum(md.Flux_xyz[:,:,0])













# Iterate MC for various settings
Tstart = time.time()
fig1, ax1 = plt.subplots(); plt.ion()  # 2D cross section
# for r_aperture in [0.00075,1]:
for gausshalfangledeg in [4,8,16,32]:
    md = mcml.MCMLModel(modelName)
    md.do_one_run(numPhotons, srcShape, r_aperture, cdfResolution, gausshalfangledeg)
    Tend = (time.time()-Tstart)
    print('r_aperture=%.5f, srcShape=%i' %(r_aperture, srcShape))
    print('--------------------------------Elapsed time: %.2f minutes--------------------------------' %(Tend/60))

    dz = md.dz
    nz = md.nz
    dr = md.dr
    nr = md.nr
    ixy_cent = md.ixy_cent
    ax1.semilogy(np.arange(0,dz*nz,dz)*1e4, md.Flux_xyz[ixy_cent][ixy_cent]/numPhotons*LEDarea/dr**2, label=gausshalfangledeg)
    
    saveName = 'mcml_scattering_nph1e6_glass2retina_rAperture15um_gaussianHA%ideg.npy' %(gausshalfangledeg)
    np.save(saveName,[numPhotons,modelName,srcShape,md])

ax1.set_title('Aperture 15um Diameter')
ax1.set_ylim(1e-3,2)
ax1.legend()

# figName1 = 'led15by15um_gap7um_lightSrcComparison_15umAperture'
# fig1.savefig('sim results\\'+figName1+'.svg', format='svg', dpi=1200)
# fig1.savefig('sim results\\'+figName1+'.png', format='png', dpi=600)
# figName2 = 'led15by15um_gap7um_lightSrcComparison_noAperture'
# fig2.savefig('sim results\\'+figName2+'.svg', format='svg', dpi=1200)
# fig2.savefig('sim results\\'+figName2+'.png', format='png', dpi=600)


# save data to npy
# np.save('mcml_scattering_nph1e6_glass2retina_rAperture15um_gaussianHA8deg--------.npy',[numPhotons,modelName,srcShape,md])
# np.save('Lambertian_nph1e7_glass2retina_rAperture15um.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('GaussHA8deg_nph1e6_glass2retina_rAperture15um.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('Collimated_nph1e6_glass2retina_rAperture15um.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_nph5e5_glass2retina_rAperture15um.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_15by19um_nph1e6_glass2retina_noAperture.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_20by34um_nph1e6_glass2retina_noAperture.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_15by19um_nph1e7_glass2retina_noAperture.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_20by34um_nph1e7_glass2retina_noAperture.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])
# np.save('MeasBeam_6by11um_nph1e7_glass2retina_noAperture.npy',[numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea])






'''
---------------------------------------------Plot Results---------------------------------------------
'''
# Load data
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('Lambertian_nph1e7_glass2retina_rAperture15um.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('GaussHA8deg_nph1e6_glass2retina_rAperture15um.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('Collimated_nph1e6_glass2retina_rAperture15um.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('MeasBeam_nph1e7_glass2retina_noAperture.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('MeasBeam_15by19um_nph1e6_glass2retina_noAperture.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('MeasBeam_15by19um_nph1e7_glass2retina_noAperture.npy', allow_pickle=True)
# [numPhotons,modelName,srcShape,gausshalfangledeg,md,r_aperture,LEDarea] = np.load('MeasBeam_6by11um_nph1e7_glass2retina_noAperture.npy', allow_pickle=True)



'''functions'''
# average irradiance over all voxels within r<sqrt((x-x0)**2+(y-y0)**2) of soma located at [x,y] coordinates.
# takes a 2D array as input, so no z-averaging involved.
def avg_over_soma(arr2d,ix_spot,iy_spot,rlim,dr):
    dum = 0
    nVox = 0
    irlim = int(rlim/dr)
    for ix in range(max(ix_spot-irlim-1,0),min(ix_spot+irlim+1,np.shape(arr2d)[0])):
        for iy in range(max(iy_spot-irlim-1,0),min(iy_spot+irlim+1,np.shape(arr2d)[1])):
            if (ix_spot-ix)**2+(iy_spot-iy)**2 < (rlim/dr)**2:
                dum += arr2d[ix,iy]
                nVox += 1
    return dum/nVox


# load coordinate settings
dz = md.dz
nz = md.nz
dr = md.dr
nr = md.nr
ixy_cent = md.ixy_cent


# crop out first 10um of multilayer stack consisting of glass-retina, to zero z at surface of retina
dz_crop = 10*1e-4  # cm
iz_crop = int(dz_crop/dz)
Flux_crop = md.Flux_xyz[:,:,iz_crop:]  #cropped flux


# attenuation along beam center
fig = plt.figure(5); plt.ion()  
plt.semilogy(np.arange(0,dz*(nz-iz_crop),dz)*1e4, Flux_crop[ixy_cent,ixy_cent,:]/numPhotons*LEDarea/dr**2, label='LED-pinhole on retina')
plt.ylim([1e-3,2])
plt.xlim([0,100])
plt.title('Pinhole 7um from medium')
# plt.legend()
plt.xlabel('Distance from LED (um)'); plt.ylabel('Irradiance attenuation factor')
plt.grid(which='both')
plt.show()
# figName = 'nph1e7_led15by15um_gap7um_aperture7um_crop'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# Beam 2D cross section
fig = plt.figure(6); plt.ion()  
aa = 0
centervalue = []
for iz in np.arange(5,30,5):
    aa += 1
    fig.add_subplot(1,5,aa)
    dat = Flux_crop[:,:,iz]/numPhotons*LEDarea/dr**2
    im = plt.imshow(dat, extent=[-dr*nr/2*1e4,dr*nr/2*1e4, -dr*nr/2*1e4,dr*nr/2*1e4], norm=colors.LogNorm())
    # im = plt.imshow(md.Flux_xyz[md.ixy_cent], norm=colors.LogNorm())
    plt.colorbar(im)
    plt.clim(0.001,1)
    plt.xlabel('um'); plt.ylabel('um')
    plt.xlim(-25,25)
    plt.ylim(-25,25)
    plt.title('Z = %ium' %int(iz*dz*1e4))
    print('%.3f' %dat[ixy_cent,ixy_cent])
    centervalue.append(dat[ixy_cent,ixy_cent])
plt.show()
# figName = 'nph1e7_led15by15um_gap7um_aperture7um_Zslices_crop'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# Beam 1D cross section
fig = plt.figure(7); plt.ion()
centervalue = []
for iz in np.arange(5,30,5):
    dat = Flux_crop[:,ixy_cent,iz]/numPhotons*LEDarea/dr**2
    plt.semilogy(np.arange(-ixy_cent,ixy_cent+1)*dr*1e4, dat, label='z=%ium' %(iz*dz*1e4))
    # im = plt.imshow(md.Flux_xyz[md.ixy_cent], norm=colors.LogNorm())
plt.grid(which='both')
plt.title('')
plt.xlabel('um'); plt.ylabel('Irradiance')
plt.legend()
plt.show()
# figName = 'nph1e7_led15by15um_gap7um_aperture7um_Zslices_1D_crop'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)







# # Average irradiance over soma cross sectional area, for z and xy offset (without interp2 smoothing)
# flux_sc = Flux_crop/numPhotons*LEDarea/dr**2  # flux scaled to LEDirradiance=1
# rsoma = 0.00075  #cm; soma radius----------------Alan, define soma size here

# fig = plt.figure(8); plt.ion()
# flux_dum = flux_sc
# irrOnSoma = np.zeros(np.shape(flux_dum)[0])
# for iz in [5,10,20,30,50]:
#     for ix in np.arange(np.shape(flux_dum)[0]):
#         dat = flux_dum[:,:,iz]
#         irrOnSoma[ix] = avg_over_soma(dat,ix,ixy_cent,rsoma,dr)
#     plt.semilogy(np.arange(-ixy_cent,ixy_cent+1)*dr*1e4, irrOnSoma, label='z=%ium' %(iz*dz*1e4))
# plt.title('Gaussian HA 8deg')
# plt.xlabel('Lateral offset of soma')
# plt.ylabel('Average Irradiance on soma')
# plt.legend()
# plt.grid(which='both')
# plt.xlim(-25,25); plt.ylim(1e-3,2)
# plt.show()
# figName = 'nph1e6_glass2retina_rAperture15um_gaussianHA8deg_zoom_crop'
# # plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# # plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# Average irradiance over soma cross sectional area, for z and xy offset
# 2D interp irrOnSoma to produce irrOnSoma_new(z,r) with finer mesh
flux_sc = Flux_crop/numPhotons*LEDarea/dr**2  # flux scaled to LEDirradiance=1
rsoma = 0.00075  #cm; soma radius----------------Alan, define soma size here

irrOnSoma = np.zeros([np.shape(flux_sc)[0],np.shape(flux_sc)[2]])
for iz in np.arange(np.shape(flux_sc)[1]):
    for ix in np.arange(np.shape(flux_sc)[0]):
        dat = flux_sc[:,:,iz]
        irrOnSoma[ix,iz] = avg_over_soma(dat,ix,ixy_cent,rsoma,dr)

raxis = np.linspace(-dr*ixy_cent,dr*ixy_cent,np.shape(irrOnSoma)[0])*1e4
zaxis = np.arange(0,dz*(nz-iz_crop),dz)*1e4
f = scipy.interpolate.interp2d(zaxis, raxis, irrOnSoma, kind='cubic')

rnew = np.arange(-dr*ixy_cent,dr*(ixy_cent+1e-6),dr/10)*1e4  # um
znew = np.arange(0,dz*(nz-iz_crop),dz/10)*1e4  # um
irrOnSoma_new = f(znew,rnew)

fig = plt.figure(9); plt.ion()
im = plt.imshow(irrOnSoma_new, extent=[znew[0],znew[-1],rnew[0],rnew[-1]], norm=colors.LogNorm())
plt.colorbar(im)
plt.clim(1e-4,1)
plt.title('Gaussian HA 8deg')
plt.xlabel('Distance from device to soma center (um)'); plt.ylabel('Lateral offset of soma(um)')
plt.show()
# figName = 'nph1e6_glass2retina_rAperture15um_gaussianHA8deg_interp2'
# plt.savefig('sim results\\'+figName+'.svg', format='svg', dpi=1200)
# plt.savefig('sim results\\'+figName+'.png', format='png', dpi=600)


# To print value of average irradiance on soma centered at [r,z] coordinates
loc_um = [0,20]     #um; (r,z) coordinates of cell location, where device passivation is at (0,0)
irnew = min(range(len(rnew)), key=lambda i: abs(rnew[i]-loc_um[0]))
iznew = min(range(len(znew)), key=lambda i: abs(znew[i]-loc_um[1]))

print('Average LED irradiation received by a 15um diameter soma at (r,z)=(%.1f,%.1f)um  is:  %.3f' %(loc_um[0],loc_um[1],irrOnSoma_new[irnew,iznew]))

