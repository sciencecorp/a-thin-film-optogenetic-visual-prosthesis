import matplotlib.pyplot as plt
import numpy as np
import csv
import math


#Import action spectra
csvname = 'ICNIRP_table2_actionSpectra.csv'
with open(csvname, newline='') as csvfile:
    dat = csv.reader(csvfile, delimiter=',', quotechar='|')
    array = list(dat)
names = array[0] #legend

actionSpec = np.genfromtxt(csvname, delimiter=',',skip_header=1)
#dimensionless; action spectra A, B, or R, a.k.a. weighting function.
#ex) actionSpec[:,0] = lam
#ex) actionSpec[:,1] = A; blue photochemical aphakic
#ex) actionSpec[:,2] = B; blue photochemical
#ex) actionSpec[:,3] = R; thermal

lam = actionSpec[:,0] #nm; wavelength
dLam = lam[1]-lam[0]
plt.semilogy(lam,actionSpec[:,1], marker="o", alpha=0.5, label='A: aphakic blue')
plt.semilogy(lam,actionSpec[:,2], marker="s", alpha=0.5, label='B: phakic blue')
plt.semilogy(lam,actionSpec[:,3], marker="^", alpha=0.5, label='R: thermal')
plt.legend()
plt.show()


#Make Gaussian LED power distribution; unit is nm-1
def ledGauss(lam,lampk,fwhm):
    dLam = lam[1]-lam[0] #nm; lambda step size
    sig = fwhm/2.355
    gauss = 1/(sig*math.sqrt(2*math.pi))*np.exp(-(lam-lampk)**2/(2*sig**2))
    gauss[gauss<1e-12]=0 #zero numbers that are small
    gauss = gauss/sum(gauss)/dLam #normalize so that sum(gauss*dLam)=1
    return gauss

#define LED spectrum
srcSpec = ledGauss(lam,500,20) #nm-1; Gaussian LED spectrum. Normalized.
plt.semilogy(lam,srcSpec)
plt.show()


#Exposure limits
#1. Thermal 380-1400nm
#single LED or LED array (both amax<a), long term exposure (0.25s<t)
L_R_EL = 2.8e5  #W m-2 sr-1; source radiance, thermal exposure limit
tau = 0.9  #dimensionsless; transmittance of ocular medium
d_p = 7e-3  #m; pupil diameter
E_R_EL = 2700 * L_R_EL * tau * d_p**2  #W m-2; retinal irradiance
E_R_EL_mWcm2 = E_R_EL * 1e-1  #mW cm-2; retinal irradiance

#2. Photochemical 300-700nm
#single LED or LED array (both amax<a), long term exposure (1e4s<t)
L_B_EL = 100  #W m-2 sr-1; source radiance, photochemical exposure limit
E_B_EL = 2700 * L_B_EL * tau * d_p**2  #W m-2; retinal irradiance
E_B_EL_mWcm2 = E_B_EL * 1e-1  #mW cm-2; retinal irradiance


#Biological effective radiance limits
fwhm = 20  #nm; full width at half maximum of Gaussian source spectrum
K_R = [0]*len(lam)
K_A = [0]*len(lam)
for ll in range(len(lam)):
    srcSpec = ledGauss(lam,lam[ll],fwhm) #nm-1; Gaussian LED spectrum. Normalized.
    K_R[ll] = np.nansum(srcSpec*actionSpec[:,3])*dLam  #thermal weighting factor
    K_A[ll] = np.nansum(srcSpec*actionSpec[:,1])*dLam  #photochemical weighting factor

plt.semilogy(lam,1/actionSpec[:,3])
plt.semilogy(lam,np.reciprocal(K_R))
plt.xlim([380+fwhm/2, 1400-fwhm/2])
plt.ylim([.5e0, 2e2])
plt.xlabel('LED center wavelength (nm)')
plt.grid(which='both')
plt.legend(['1/R($\lambda$)','1/$\kappa_R$'])
plt.show()

plt.semilogy(lam,1/actionSpec[:,1])
plt.semilogy(lam,np.reciprocal(K_A))
plt.xlim([300+fwhm/2, 700-fwhm/2])
plt.ylim([.5e-1, 2e3])
plt.xlabel('LED center wavelength (nm)')
plt.grid(which='both')
plt.legend(['1/A($\lambda$)','1/$\kappa_A$'])
plt.show()

plt.semilogy(lam,E_R_EL_mWcm2 * np.reciprocal(K_R))
plt.semilogy(lam,E_B_EL_mWcm2 * np.reciprocal(K_A))
plt.xlim([300+fwhm/2, 700-fwhm/2])
plt.ylim([.5e-1, 2e5])
plt.xlabel('LED center wavelength (nm)')
plt.ylabel('Max. Irradiance ($\mathrm{mW/cm^2}$)')
plt.grid(which='both')
plt.legend(['Thermal Exposure Limit','Photochemical Exposure Limit'])
plt.title('Permitted LED Irradiance for Retinal Implant')
plt.show()


