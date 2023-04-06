import numpy as np
from Bresenham3D import *

WEIGHT = 1e-4       # critical weight for roulette
CHANCE = 0.1		    # Chance of roulette survival
PARTIALREFLECTION = 0     # 1=split photon, 0=statistical reflection.
COSZERO = 1.0 - 1.0e-12     # cosine of about 1e-6 rad
COS90D = 1.0e-6     # cosine of about 1.57 - 1e-6 rad
verbose = 0         #status updates
savetraj = 0        #save trajectory coordinates

class Medium:
    """Medium class - optical medium class defining the optical properties
        Class instance variables:
            n - refractive index
            mua - absorption coefficient. [1/cm]
            mus - scattering coefficient. [1/cm]
            g - anisotropy
        Methods:
    """
    
    def __init__(self, mediumName = 'DERMIS'):
        
        # initialize medium optical properties (mua, mus in units cm-1)
        if mediumName.lower() == 'AIR'.lower():
            self.n = 1.0
            self.mua = 0.0
            self.mus = 0.0
            self.g = 1.0
        elif mediumName.lower() == 'GLASS'.lower():    # 800-nm wavelngth
            self.n = 2.0
            self.mua = 0.00001
            self.mus = 0.00001
            self.g = 1.0
        elif mediumName.lower() == 'DERMIS'.lower():    # 800-nm wavelngth
            self.n = 1.4
            self.mua = 0.26
            self.mus = 137.0
            self.g = 0.90
        elif mediumName.lower() == 'TYPE_II_EPIDERMIS'.lower(): # 800-nm
            self.n = 1.3
            self.mua = 5.0
            self.mus = 200.0
            self.g = 0.70
        elif mediumName.lower() == 'CORNEA'.lower():    # 1060-nm
            self.n = 1.376
            self.mua = 0.157    # from Bovine NIR paper 2011
            self.mus = 1.064
            self.g = 0.90
        elif mediumName.lower() == 'AQUEOUS_HUMOR'.lower():    # 1060-nm
            self.n = 1.336
            self.mua = 0.78     # from Optical properties of ocular tissues
            self.mus = 0.60
            self.g = 0.99
        elif mediumName.lower() == 'LENS'.lower():    # 1060-nm
            self.n = 1.406
            self.mua = 0.206     # from Bovine NIR paper 2011
            self.mus = 1.131
            self.g = 0.90
        elif mediumName.lower() == 'VITREOUS_HUMOR'.lower():    # 1060-nm
            self.n = 1.337
            self.mua = 0.48     # from Optical properties of ocular tissues
            self.mus = 0.25
            self.g = 0.99
        elif mediumName.lower() == 'RETINA'.lower():    # 1060-nm
            self.n = 1.358
            self.mua = 2.745     # from Bovine NIR paper 2011
            self.mus = 71.50
            self.g = 0.70
        else:
            self.n = 1.4
            self.mua = 0.26
            self.mus = 137.0
            self.g = 0.90     
        

class LayerStruct:
    """LayerStruct class - multi-layered structure
        Class instance variables:
            nIn - refractive index of the incidence medium
            nOut - refractive index of the exit medium
            numLayers - number of layers
            layer - list of layer objects
            layerThickness - layer thickness in [cm]
            layerZ - layer depth z coordinates, top and bottom [cm]
            cosCrit - ciritical angle cosines of each layer, top and bottom
        Methods:
            
    """
    
    def __init__(self, layerName = 'BARE_DERMIS'):
        if layerName.lower() == 'BARE_DERMIS'.lower():
            self.numLayers = 1
            self.layer = [Medium('AIR'), Medium('DERMIS'), Medium('DERMIS')]
            self.layerThickness = [0.3]
        elif layerName.lower() == 'TYPE_II_SKIN'.lower():
            self.numLayers = 2  # number of layers
            self.layer = [Medium('AIR'), Medium('TYPE_II_EPIDERMIS'), \
                Medium('DERMIS'), Medium('DERMIS')]
            self.layerThickness = [0.006, 0.3]    # in [cm]
        elif layerName.lower() == 'CORNEA'.lower():
            self.numLayers = 1  # number of layers
            self.layer = [Medium('AIR'), Medium('CORNEA'), \
                Medium('AQUEOUS_HUMOR')]
            self.layerThickness = [0.0449]    # in [cm]
        elif layerName.lower() == 'EYE_ANTERIOR'.lower():
            self.numLayers = 3  # number of layers
            self.layer = [Medium('AIR'), Medium('CORNEA'), \
                Medium('AQUEOUS_HUMOR'), Medium('LENS'), \
                Medium('VITREOUS_HUMOR')]
            self.layerThickness = [0.0449, 0.2794, 0.4979]    # in [cm]
        elif layerName.lower() == 'RETINA'.lower():
            self.numLayers = 2  # number of layers
            self.layer = [Medium('VITREOUS_HUMOR'), Medium('VITREOUS_HUMOR'), Medium('RETINA'), Medium('RETINA')]
            self.layerThickness = [0.0100,0.1000]    # in [cm]
        elif layerName.lower() == 'TEST'.lower():
            self.numLayers = 2  # number of layers
            self.layer = [Medium('VITREOUS_HUMOR'), Medium('VITREOUS_HUMOR'), Medium('RETINA'), Medium('RETINA')]
            self.layerThickness = [0.0007, 0.2000]    # in [cm]
            self.zled = self.layerThickness[0]  #JC; where LED
        elif layerName.lower() == 'LEDSTACK'.lower():
            self.numLayers = 2  # number of layers
            self.layer = [Medium('GLASS'), Medium('GLASS'), Medium('RETINA'), Medium('RETINA')]
            self.layerThickness = [0.0007, 0.2000]    # in [cm]
            self.zled = self.layerThickness[0]  #JC; where LED
        else:
            self.numLayers = 1
            self.layer = [Medium('AIR'), Medium('DERMIS'), Medium('DERMIS')]
            self.layerThickness = [0.3]
        self.layerZ = []
        self.cosCrit = []
        z = 0   # incidence first medium z coordinate [cm]
        self.layerZ.append([0, 0])  # first incidence medium, not used
        self.cosCrit.append([0, 0])  # first incidence medium, not used
        # find the z depth coordinates and cosine critical angles for each
        #   layer
        for i in range(1, self.numLayers+1):
            self.layerZ.append([z, z+self.layerThickness[i-1]])
            z = self.layerZ[-1][1]
            # calculate the critical angle cosines for each layer
            # crticial angle at top interface of the current layer
            n1 = self.layer[i].n
            n2 = self.layer[i-1].n
            if n1 > n2:
                cosCrit0 = (1.0 - n2*n2/(n1*n1))**0.5
            else:
                cosCrit0 = 0.0
            # crticial angle at bottom interface of the current layer
            n2 = self.layer[i+1].n
            if n1 > n2:
                cosCrit1 = (1.0 - n2*n2/(n1*n1))**0.5
            else:
                cosCrit1 = 0.0
            self.cosCrit.append([cosCrit0, cosCrit1])

        if verbose: #JC
            print(self.layerZ)
            print(self.cosCrit)

    def calc_r_specular(self):
        # direct reflections from the 1st and 2nd layers.
        temp = (self.layer[0].n - self.layer[1].n)/(self.layer[0].n + \
            self.layer[1].n)
        r1 = temp*temp
  
        if ((self.layer[1].mua == 0.0) and (self.layer[1].mus == 0.0)):
            # glass layer.
            temp = (self.layer[1].n - self.layer[2].n)/(self.layer[1].n + \
                self.layer[2].n)
            r2 = temp*temp
            r1 = r1 + (1 - r1)*(1 - r1)*r2/(1 - r1*r2) 
        return r1


class ModelInput:
    """ModelInput class - multi-layered photon scattering model input
        Class instance variables:
            Wth - play roulette if photon weight < Wth
            dz - z grid separation [cm]
            dr - r grid separation [cm]
            da - alpha grid separation [radian]
            nz - array range 0..nz-1
            nr - array range 0..nr-1
            na - array range 0..na-1
            layerObj - medium layer structure class instance
        Methods:
            
    """
    
    def __init__(self, modelName = 'BARE_DERMIS'):
        if modelName.lower() == 'BARE_DERMIS'.lower():
            self.layerObj = LayerStruct('BARE_DERMIS')
            self.dz = 100e-4
            self.dr = 100e-4
            self.nz = 30
            self.nr = 50
            self.na = 10
        elif modelName.lower() == 'TYPE_II_SKIN'.lower():
            self.layerObj = LayerStruct('TYPE_II_SKIN')
            self.dz = 20e-4
            self.dr = 20e-4
            self.nz = 10
            self.nr = 50
            self.na = 10
        elif modelName.lower() == 'CORNEA'.lower():
            self.layerObj = LayerStruct('CORNEA')
            self.dz = 10e-4
            self.dr = 10e-4
            self.nz = 100
            self.nr = 50
            self.na = 10
        elif modelName.lower() == 'EYE_ANTERIOR'.lower():
            self.layerObj = LayerStruct('EYE_ANTERIOR')
            self.dz = 20e-4
            self.dr = 20e-4
            self.nz = 500
            self.nr = 250
            self.na = 10 
        elif modelName.lower() == 'RETINA'.lower():
            self.layerObj = LayerStruct('RETINA')
            self.dz = 5e-4  # cm
            self.dr = 5e-4  # cm
            self.nz = 40
            self.nr = 40
            self.na = 10 
        elif modelName.lower() == 'TEST'.lower():
            self.layerObj = LayerStruct('TEST')
            self.dz = 0.0002  # cm
            self.dr = 0.0002  # cm
            self.nz = 100
            self.nr = 100
            self.na = 10 
        elif modelName.lower() == 'LEDSTACK'.lower():
            self.layerObj = LayerStruct('LEDSTACK')
            self.dz = 0.0002  # cm
            self.dr = 0.0002  # cm
            self.nz = 100
            self.nr = 100
            self.na = 10 
        else:
            self.layerObj = LayerStruct('BARE_DERMIS')
            self.dz = 100e-4
            self.dr = 100e-4
            self.nz = 30
            self.nr = 50
            self.na = 10
        self.Wth = WEIGHT
        self.da = 0.5*np.pi/self.na

class MCMLModel(ModelInput):
    """MCMLModel class - multi-layered photon scattering model, inherits from
        ModelInput layer structure setup
        Class instance variables:
            Rsp - specular reflectance [-]
            Rd - total diffuse reflectance [-]
            A - total absorption probability [-]
            Tt - total transmittance [-]
            Rd_ra - 2D distribution of diffuse reflectance [1/(cm2 sr)]
            Rd_r - 1D radial distribution of diffuse reflectance [1/cm2]
            Rd_a - 1D angular distribution of diffuse reflectance [1/sr]
            A_rz - 2D probability density in turbid media over r & z [1/cm3]
            A_xyz - 3D probability density in turbid media over x, y & z [1/cm3]  #JC
            Flux_xyz - 3D probability density in turbid media over x, y & z [1/cm3]  #JC
            A_z - 1D probability density over z [1/cm]
            A_l - each layer's absorption probability [-]
            Tt_ra - 2D distribution of total transmittance [1/(cm2 sr)]
            Tt_r - 1D radial distribution of transmittance [1/cm2]
            Tt_a - 1D angular distribution of transmittance [1/sr]
        Methods:
            
    """

    def __init__(self, modelName = 'BARE_DERMIS'):
        # extend the ModelInput base class instance variables
        ModelInput.__init__(self, modelName)
        self.numPhotons = 0
        # initialize the model grid arrays    
        self.Rsp = self.layerObj.calc_r_specular()
        self.Rd = 0.0
        self.A = 0.0
        self.Tt = 0.0
        self.Rd_ra = np.matrix(np.zeros((self.nr, self.na)))
        self.Rd_r = np.zeros(self.nr)
        self.Rd_a = np.zeros(self.na)
        self.A_rz = np.matrix(np.zeros((self.nr, self.nz)))
        self.A_xyz = np.zeros((self.nr+1, self.nr+1, self.nz))  #JC
        self.ixy_cent = int(self.nr/2)  #JC
        self.Flux_xyz = np.zeros((self.nr+1, self.nr+1, self.nz))  #JC
        self.A_z = np.zeros(self.nz)
        self.A_l = np.zeros(2 + self.layerObj.numLayers)
        self.Tt_ra = np.matrix(np.zeros((self.nr, self.na)))
        self.Tt_r = np.zeros(self.nr)
        self.Tt_a = np.zeros(self.na)

        if savetraj: self.traj = []  #JC

    def do_one_run(self, numPhotons, srcShape, LEDx, LEDy, r_aperture, cdfResolution, gausshalfangledeg):
        GaussCDF = self.GaussSinCDF(cdfResolution, gausshalfangledeg)  # cumulative distribution function of Gaussian function
        MeasBeamCDF = self.MeasBeamSinCDF(cdfResolution)  # cumulative distribution function of Gaussian function
        self.deadByAperture = 0
        for i in range(numPhotons):
            photon = Photon(srcShape, LEDx, LEDy, r_aperture, self.layerObj, self.Rsp, GaussCDF, MeasBeamCDF, cdfResolution, self)

            if verbose: print('\nLaunch photon #%i          in direction [%.3f,%.3f,%.3f]' %(i,photon.ux,photon.uy,photon.uz))

            photon.run_one_photon(self)

            if np.mod(i,1000)==0:
                print(i,i/numPhotons)

            if savetraj:  self.traj = photon.traj
        print('number of photons blocked by aperture = %i' %self.deadByAperture)

    def GaussSinCDF(self, numPoints, gausshalfangledeg): # Cumulative Distribution Function generator of Gaussian*Sine function
        # Gaussian distribution
        theta = np.linspace(0,np.pi/2,numPoints+1)
        fwhm = (gausshalfangledeg*2)/180*np.pi
        sig = fwhm/(2*np.sqrt(2*np.log(2)))
        gaussDist = 2 * 1/(sig*np.sqrt(2*np.pi)) * np.exp(-np.multiply(theta*theta, 1/(2*sig**2)))

        # Gaussian times sine, for Spherical probability distribution
        gaussSine = np.multiply(gaussDist,np.sin(theta))
        curveInteg = sum(gaussSine)*(np.pi/2/numPoints)
        gaussSineNorm = gaussSine/curveInteg
        curveInteg = sum(gaussSineNorm)*(np.pi/2/numPoints)

        # Gaussian*sine CDF
        cdf = np.cumsum(gaussSineNorm)*(np.pi/2/numPoints)
        return cdf

    def MeasBeamSinCDF(self, numPoints): # Cumulative Distribution Function generator of Gaussian*Sine function
        # Beam profile
        [theta, measBeamDat] = np.load('FlexLED03_B1_smoothed.npy')
        iend = np.where(theta==90)[0][0]
        theta = theta[:iend]/180*3.14159
        measBeamDat = measBeamDat[:iend]

        # Interpolate
        theta2 = np.linspace(0,np.pi/2,numPoints+1)
        measBeamDat_interp = np.interp(theta2,theta,measBeamDat)

        # Beam profile times sine, for Spherical probability distribution
        beamSine = np.multiply(measBeamDat_interp,np.sin(theta2))
        curveInteg = sum(beamSine)*(np.pi/2/numPoints)
        beamSineNorm = beamSine/curveInteg
        curveInteg = sum(beamSineNorm)*(np.pi/2/numPoints)

        # Beam*sine CDF
        cdf = np.cumsum(beamSineNorm)*(np.pi/2/numPoints)
        return cdf

    def sum_scale_result(self):
        # Get 1D & 0D results.
        self.sum_2D_Rd()
        self.sum_2D_A()
        self.sum_2D_Tt()  
        self.scale_Rd_Tt()
        self.scale_A()

    def sum_2D_Rd(self):
        #	Get 1D array elements by summing the 2D array elements.
        for ir in range(self.nr):
            sum = 0.0
            for ia in range(self.na):
                sum += self.Rd_ra[ir, ia]
            self.Rd_r[ir] = sum
  
        for ia in range(self.na):
            sum = 0.0
            for ir in range(self.nr):
                sum += self.Rd_ra[ir, ia]
            self.Rd_a[ia] = sum
  
        sum = 0.0
        for ir in range(self.nr):
            sum += self.Rd_r[ir]
        self.Rd = sum

    def iz_to_layer(self, iz):
        # Return the index to the layer according to the index
        # to the grid line system in z direction (Iz).
        # Use the center of box.
        i = 1     	# index to layer.
        while ((iz+0.5)*self.dz >= self.layerObj.layerZ[i][1] and \
            i < self.layerObj.numLayers):
            i += 1
  
        return i

    def sum_2D_A(self):
        # Get 1D array elements by summing the 2D array elements.
        for iz in range(self.nz):
            sum = 0.0
            for ir in range(self.nr):
                sum += self.A_rz[ir, iz]
            self.A_z[iz] = sum
  
        sum = 0.0
        for iz in range(self.nz):
            sum += self.A_z[iz]
            self.A_l[self.iz_to_layer(iz)] += self.A_z[iz]

        self.A = sum

    def sum_2D_Tt(self):     
        # Get 1D array elements by summing the 2D array elements.
  
        for ir in range(self.nr):
            sum = 0.0
            for ia in range(self.na):
                sum += self.Tt_ra[ir, ia]
            self.Tt_r[ir] = sum
  
        for ia in range(self.na):
            sum = 0.0
            for ir in range(self.nr):
                sum += self.Tt_ra[ir, ia]
            self.Tt_a[ia] = sum
  
        sum = 0.0
        for ir in range(self.nr):
            sum += self.Tt_r[ir]
        self.Tt = sum            

    def scale_Rd_Tt(self):
        # Scale Rd and Tt properly.
        # "a" stands for angle alpha.
        # Scale Rd(r,a) and Tt(r,a) by
        # (area perpendicular to photon direction)
        # x(solid angle)x(No. of photons).
        # or
        # [2*PI*r*dr*cos(a)]x[2*PI*sin(a)*da]x[No. of photons]
        # or
        # [2*PI*PI*dr*da*r*sin(2a)]x[No. of photons]
        # Scale Rd(r) and Tt(r) by
        # (area on the surface)x(No. of photons).
        # Scale Rd(a) and Tt(a) by
        # (solid angle)x(No. of photons).

        scale1 = 4.0*np.pi*np.pi*self.dr*np.sin(self.da/2)\
          *self.dr*self.numPhotons

        # The factor (ir+0.5)*sin(2a) to be added.

        for ir in range(self.nr):  
            for ia in range(self.na):
                scale2 = 1.0/((ir+0.5)*np.sin(2.0*(ia+0.5)*self.da)*scale1)
                self.Rd_ra[ir, ia] *= scale2
                self.Tt_ra[ir, ia] *= scale2
  
        scale1 = 2.0*np.pi*self.dr*self.dr*self.numPhotons  
        # area is 2*PI*[(ir+0.5)*dr]*dr. 
        # ir+0.5 to be added. */

        for ir in range(self.nr):
            scale2 = 1.0/((ir+0.5)*scale1)
            self.Rd_r[ir] *= scale2
            self.Tt_r[ir] *= scale2
  
        scale1  = 2.0*np.pi*self.da*self.numPhotons
        # solid angle is 2*PI*sin(a)*da. sin(a) to be added.

        for ia in range(self.na):
            scale2 = 1.0/(np.sin((ia+0.5)*self.da)*scale1)
            self.Rd_a[ia] *= scale2
            self.Tt_a[ia] *= scale2
  
        scale2 = 1.0/self.numPhotons
        self.Rd *= scale2
        self.Tt *= scale2

    def scale_A(self):
        # Scale absorption arrays properly.
        # Scale A_rz.
        scale1 = 2.0*np.pi*self.dr*self.dr*self.dz*self.numPhotons	
        # volume is 2*pi*(ir+0.5)*dr*dr*dz.*/ 
        # ir+0.5 to be added.
        for iz in range(self.nz):
            for ir in range(self.nr):
                self.A_rz[ir, iz] /= (ir+0.5)*scale1

        # Scale A_xyz. #JC
        scale1 = self.dr*self.dr*self.dz*self.numPhotons	
        # volume is dx*dy*dz.*/ 
        for iz in range(self.nz):
            for ix in range(self.nr+1):
                for iy in range(self.nr+1):
                    self.A_xyz[ix, iy, iz] /= scale1
  
        # Scale A_z.
        scale1 = 1.0/(self.dz*self.numPhotons)
        for iz in range(self.nz):
            self.A_z[iz] *= scale1
  
        # Scale A_l. Avoid int/int.
        scale1 = 1.0/self.numPhotons	
        for il in range(self.layerObj.numLayers+2):
            self.A_l[il] *= scale1
  
        self.A *=scale1

    def get_mua_at_iz(self, iz):
        i = 1       # index to layer
        numLayers = self.layerObj.numLayers
        dz = self.dz
        while ((iz + 0.5)*dz >= self.layerObj.layerZ[i][1] and i < numLayers):
            i += 1
        mua = self.layerObj.layer[i].mua
        return mua

 

class Photon:
    """Photon class - MCML photon class for Monte Carlo scattering model in
        multilayered turbid media. 
        Class instance variables:
            x = Cartesian coordinate x [cm]
            y = Cartesian coordinate y [cm]
            z = Cartesian coordinate z [cm]
            ux = directional cosine x of a photon
            uy = directional cosine y of a photon
            uz = directional cosine z of a photon
            w - weight
            dead - true if photon is terminated
            layer - index to layer where the photon packet resides
            s - current step size [cm]
            sleft - step size left, dimensionless [-]           
        Methods:
            
    """

    def __init__(self, srcShape=1, LEDx=0.0015, LEDy=0.0015, r_aperture=1, layerObj=LayerStruct('BARE_DERMIS'),
        rSpecular=0.0, GaussCDF=0, MeasBeamCDF=0, cdfResolution=0, model=0):

        # initialize a photon
        # self.x = 0.0
        # self.y = 0.0
        # self.z = 0.0

        # LEDx = 0.0015  # LED x and y width
        # LEDy = 0.0019  # LED x and y width
        self.x = np.random.random_sample()*LEDx - LEDx/2
        self.y = np.random.random_sample()*LEDy - LEDy/2
        self.z = 0.0

        if savetraj: self.traj = []; self.traj.append([self.x,self.y,self.z])  #JC

        if srcShape==0:  # Collimated; JC
            self.ux = 0.0
            self.uy = 0.0
            self.uz = 1.0
        elif srcShape==1:  # Lambertian; JC
            [self.ux,self.uy,self.uz] = self.startLambertian(0,0,1)
        elif srcShape==2:  # Gaussian; JC
            [self.ux,self.uy,self.uz] = self.startGaussian(0,0,1,GaussCDF,cdfResolution)
        elif srcShape==3:  # Measured; JC
            [self.ux,self.uy,self.uz] = self.startMeasBeam(0,0,1,MeasBeamCDF,cdfResolution)
            

        self.w = 1.0 - rSpecular
        self.dead = False
        self.layer = 1  # initial layer
        self.s = 0
        self.sleft = 0

        #Pinhole here!!
        self.pass_aperture(layerObj, r_aperture,model)  # kill photon if not through aperture

        # take care of the case when the first layer is glass
        if (layerObj.layer[1].mua == 0.0) and (layerObj.layer[1].mus == 0.0):
            if verbose: ('First layer is glass, jump to next layer')
            self.layer = 2      # skip to next layer
            self.z = layerObj.layerZ[2][0]  # use z0 from the next layer
        
    def run_one_photon(self, model):
        while self.dead == False:
            self.hop_drop_spin(model)
        model.numPhotons += 1

    def pass_aperture(self,layerObj,r_aperture,model):  # JC
        z2bdry = layerObj.layerZ[1][1]
        r_at_bdry_sq = (self.x + self.ux * z2bdry)**2 + (self.y + self.uy * z2bdry)**2
        if  r_at_bdry_sq > r_aperture**2 :
            self.dead = True
            model.deadByAperture += 1

    def hop_drop_spin(self, model):
        layer = self.layer
        if (model.layerObj.layer[layer].mua == 0.0) and \
            (model.layerObj.layer[layer].mus == 0.0):
            # glass layer
            self.hop_in_glass(model)
        else:
            self.hop_drop_spin_in_tissue(model)
        
        if (self.w < model.Wth) and (self.dead == False):
            self.roulette

    def hop_in_glass(self, model):
        # Move the photon packet in glass layer.
        # Horizontal photons are killed because they will
        # never interact with tissue again.
        if self.uz == 0.0:
            # horizontal photon in glass is killed
            self.dead = True
        else:
            self.step_size_in_glass(model)
            self.hop(model,self.s)
            self.cross_or_not(model)
        
        if verbose: print('hop_in_glass')

    def hop_drop_spin_in_tissue(self, model):
        # Set a step size, move the photon, drop some weight, 
        # choose a new photon direction for propagation.
        # When a step size is long enough for the photon to 
        # hit an interface, this step is divided into two steps. 
        # First, move the photon to the boundary free of 
        # absorption or scattering, then decide whether the 
        # photon is reflected or transmitted.
        # Then move the photon in the current or transmission 
        # medium with the unfinished stepsize to interaction 
        # site.  If the unfinished stepsize is still too long, 
        # repeat the above process.
        self.step_size_in_tissue(model)  # create new step size if not any left
        if self.hit_boundary(model) == True:
            if verbose: print('hop_drop_spin_in_tissue: hit boundary')
            self.hop(model,self.s)      # move to boundary plane
            self.cross_or_not(model)
        else:
            if verbose: print('hop_drop_spin_in_tissue: didnt hit boundary')
            self.hop(model,self.s)
            self.drop(model)
            self.spin(model.layerObj.layer[self.layer].g)

    def step_size_in_glass(self, model):
        # If uz != 0, return the photon step size in glass, 
        # Otherwise, return 0.
        # The step size is the distance between the current 
        # position and the boundary in the photon direction.
        # Make sure uz !=0 before calling this function.

        # Stepsize to the boundary.
        layer = self.layer
        uz = self.uz
        if uz > 0.0:
            dl_b = (model.layerObj.layerZ[layer][1] - self.z)/uz
        elif uz < 0.0:
            dl_b = (model.layerObj.layerZ[layer][0] - self.z)/uz
        else:
            dl_b = 0.0
  
        self.s = dl_b
                
        if verbose:
            print('stepSizeInGlass=%.3f' %(self.s))

    def step_size_in_tissue(self, model):
        # Pick a step size for a photon packet when it is in 
        # tissue.
        # If the member sleft is zero, make a new step size 
        # with: -log(rnd)/(mua+mus).
        # Otherwise, pick up the leftover in sleft.
        # Layer is the index to layer.
        # In_Ptr is the input parameters.

        layer = self.layer
        mua = model.layerObj.layer[layer].mua
        mus = model.layerObj.layer[layer].mus

        if self.sleft == 0.0:      # make a new step
            rnd = np.random.random_sample()
            self.s = -np.log(rnd)/(mua + mus)
        else:                       # take the leftover
            self.s = self.sleft/(mua + mus)
            self.sleft = 0.0
        
        if verbose:
            print('step_size_in_tissue:     layer=%i, new_s=%.3f, new_sleft=%.3f' %(layer,self.s,self.sleft))

    def hop(self,model,s):  #JC
        # Move the photon s away in the current layer of medium.
        # JC added: hop while performing Bresenham3D
        # s = self.s
        x = self.x
        y = self.y
        z = self.z
        self.x += s*self.ux
        self.y += s*self.uy
        self.z += s*self.uz

        if verbose: print('hop:                     hop from [%.3f,%.3f,%.3f] to [%.3f,%.3f,%.3f]. Cur layer %i, step len %.3f' %(x,y,z,self.x,self.y,self.z,self.layer,s))
        if savetraj: self.traj.append([self.x,self.y,self.z])

        self.record_flux(model,x,y,z)

    def record_flux(self,model,x,y,z): #JC
        # record rasterized path for light flux
        pt = np.around(np.divide([x,y,z,self.x,self.y,self.z],model.dz))  # path endpoints in pixellated bins
        traj_px = Bresenham3D(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5])  # pixellated trajectory in integers
        if verbose: print('record_flux:             rasterize [%.3f,%.3f,%.3f] to [%i,%i,%i]' %(self.x,self.y,self.z,pt[3],pt[4],pt[5]))
        
        # adj_weight = self.w * self.s/(len(traj_px)*model.dz)  # scale wt to conserve energy after pixellation
        adj_weight = self.w                                     # don't scale wt to conserve energy after pixellation
        for pp in traj_px[:-1] :  # don't include last bin
            ppx = int(pp[0]+model.ixy_cent); ppy = int(pp[1]+model.ixy_cent); ppz = int(pp[2])
            if (0<=ppx<model.nr and 0<=ppy<model.nr and 0<=ppz<model.nz) :
                model.Flux_xyz[ppx,ppy,ppz] += adj_weight
                if verbose: print('record_flux:             mark indices [%.3f,%.3f,%.3f]' %(ppx*model.dr,ppy*model.dr,ppz*model.dz))

        # if verbose: print('record_flux:             rasterize [%.3f,%.3f,%.3f] to [%i,%i,%i]' %(self.x,self.y,self.z,ppx,ppy,ppz))

    def cross_or_not(self, model):
        if self.uz < 0.0:
            self.cross_up_or_not(model)
        else:
            self.cross_dn_or_not(model)

    def cross_up_or_not(self, model):
        # Decide whether the photon will be transmitted or 
        # reflected on the upper boundary (uz<0) of the current 
        # layer.
        # If "layer" is the first layer, the photon packet will 
        # be partially transmitted and partially reflected if 
        # PARTIALREFLECTION is set to 1,
        # or the photon packet will be either transmitted or 
        # reflected determined statistically if PARTIALREFLECTION 
        # is set to 0.
        # Record the transmitted photon weight as reflection.
        # If the "layer" is not the first layer and the photon 
        # packet is transmitted, move the photon to "layer-1".
        # Update the photon parmameters.

        uz = self.uz
        r = 0.0     # reflectance
        layer = self.layer
        ni = model.layerObj.layer[layer].n
        nt = model.layerObj.layer[layer-1].n
  
        # Get r.
        if -uz <= model.layerObj.cosCrit[layer][0]:
            r = 1.0		      # total internal reflection
        else:
            r, uz1 = self.RFresnel(ni, nt, -uz)
  
        if PARTIALREFLECTION == 1:
            if (layer == 1) and (r < 1.0):      # partially transmitted
                self.uz = -uz1          # transmitted photon
                self.record_R(r, model)
                self.uz = -uz           # reflected photon		
            elif np.random.random_sample() > r:     # transmitted to layer-1
                self.layer -= 1
                self.ux *= ni/nt
                self.uy *= ni/nt
                self.uz = -uz1
            else	:		      		# reflected
                self.uz = -uz
        else:
            if np.random.random_sample() > r:   # transmitted to layer-1  #JC
                if layer == 1:
                    # self.uz = -uz1
                    # self.record_R(model, 0.0)
                    self.hop(model,self.sleft)
                    self.dead = True
                else:
                    self.layer -= 1
                    self.ux *= ni/nt
                    self.uy *= ni/nt
                    self.uz = -uz1
            else: 						# reflected
                self.uz = -uz
        
        if verbose: print('cross_up_or_not')

    def cross_dn_or_not(self, model):
        # Decide whether the photon will be transmitted  or be 
        # reflected on the bottom boundary (uz>0) of the current 
        # layer.
        # If the photon is transmitted, move the photon to 
        # "layer+1". If "layer" is the last layer, record the 
        # transmitted weight as transmittance. See comments for 
        # CrossUpOrNot.
        # Update the photon parmameters.
        
        uz = self.uz        # z directional cosine
        r = 0.0             # reflectance
        layer = self.layer
        ni = model.layerObj.layer[layer].n
        nt = model.layerObj.layer[layer+1].n
  
        # Get r
        if uz <= model.layerObj.cosCrit[layer][1]: 
            r=1.0		# total internal reflection
        else:
            r, uz1 = self.RFresnel(ni, nt, uz)
        
        if verbose: print('cross_dn_or_not:         layer=%i, r=%.3f, ni=%.3f, nt=%.3f, uz=%.3f' %(layer,r,ni,nt,uz))
  
        if PARTIALREFLECTION	 == 1:
            if (layer == model.layerObj.numLayers) and (r < 1.0):
                self.uz = uz1
                self.record_T(r, model)
                self.uz = -uz
            elif np.random.random_sample() > r:     # transmitted to layer+1
                self.layer += 1
                self.ux *= ni/nt
                self.uy *= ni/nt
                self.uz = uz1
            else: 						# reflected
                self.uz = -uz
        else:
            if np.random.random_sample() > r:	# transmitted to layer+1
                if layer == model.layerObj.numLayers:  # last layer
                    # self.uz = uz1
                    # self.record_T(model, 0.0)
                    self.hop(model,self.sleft)
                    self.dead = True
                    if verbose: print('cross_dn_or_not:         last layer. Record as transmission and kill photon')
                else:  # not last layer
                    self.layer += 1
                    self.ux *= ni/nt
                    self.uy *= ni/nt
                    self.uz = uz1
                    if verbose: print('cross_dn_or_not:         NOT last layer. Transmit to layer: %i' %self.layer)
            else: 						# reflected
                self.uz = -uz
                if verbose: print('cross_dn_or_not: reflected')

    def hit_boundary(self, model):
        # Check if the step will hit the boundary.
        # Return 1 if hit boundary.
        # Return 0 otherwise.
        # If the projected step hits the boundary, the members
        # s and sleft of Photon_Ptr are updated.

        layer = self.layer
        uz = self.uz
  
        # Distance left to boundary
        if uz > 0.0:
            dl_b = (model.layerObj.layerZ[layer][1] - self.z)/uz	# dl_b>0
        elif uz < 0.0:
            dl_b = (model.layerObj.layerZ[layer][0] - self.z)/uz    # dl_b>0
  
        if (uz != 0.0) and (self.s > dl_b):
            # not horizontal & crossing
            mut = model.layerObj.layer[layer].mua + \
                model.layerObj.layer[layer].mus
            if verbose: print('hit_boundary:            mut=%.3f' %(mut))
            self.sleft = (self.s - dl_b)*mut  # JC: why is distance left scaled by mut?????
            self.s = dl_b
            hit = True
        else:
            hit = False

        if verbose: print('hit_boundary:            hit=%r, d2bdry=%.3f, s=%.3f, sleft=%.3f' %(hit,dl_b,self.s,self.sleft))

        return hit

    def drop(self, model):
        # Drop photon weight inside the tissue (not glass).
        # The photon is assumed not dead.
        # The weight drop is dw = w*mua/(mua+mus).
        # The dropped weight is assigned to the absorption array 
        # elements.

        x = self.x
        y = self.y
        layer = self.layer
 
        # compute array indices
        iz = int(self.z/model.dz)
        if iz > (model.nz - 1):
            iz = model.nz - 1
  
        ir = int((x*x + y*y)**0.5/model.dr)
        if ir > (model.nr - 1):
            ir = model.nr - 1

        ix = int(self.x/model.dr)  #JC
        if ix > model.ixy_cent:
            ix = model.ixy_cent
        elif ix < -model.ixy_cent:
            ix = -model.ixy_cent
  
        iy = int(self.y/model.dr)  #JC
        if iy > model.ixy_cent:
            iy = model.ixy_cent
        elif iy < -model.ixy_cent:
            iy = -model.ixy_cent
        
        # update photon weight.
        mua = model.layerObj.layer[layer].mua
        mus = model.layerObj.layer[layer].mus
        dwa = self.w * mua/(mua+mus)
        self.w -= dwa
  
        # assign dwa to the absorption array element.
        model.A_rz[ir, iz] += dwa
        model.A_xyz[ix+model.ixy_cent, iy+model.ixy_cent, iz] += dwa   #JC start here! need to add negative ix and iy to center to zero.

        if verbose: print('drop:                    weight=%.5f' %(self.w))

    def spin(self, g):
        # Choose a new direction for photon propagation by 
        # sampling the polar deflection angle theta and the 
        # azimuthal angle psi.
        # Note:
        # theta: 0 - pi so sin(theta) is always positive 
        # feel free to use sqrt() for cos(theta). 
        # psi:   0 - 2pi 
        # for 0-pi  sin(psi) is + 
        # for pi-2pi sin(psi) is - 

        ux = self.ux
        uy = self.uy
        uz = self.uz

        cost = self.SpinTheta(g)
        sint = (1.0 - cost*cost)**0.5	
        # sqrt() is faster than sin().

        psi = 2.0*np.pi*np.random.random_sample()       # spin psi 0-2pi
        cosp = np.cos(psi)
        if psi < np.pi:
            sinp = (1.0 - cosp*cosp)**0.5
            # sqrt() is faster than sin().
        else:
            sinp = -(1.0 - cosp*cosp)**0.5	
  
        if np.fabs(uz) > COSZERO:       # normal incident.
            self.ux = sint*cosp
            self.uy = sint*sinp
            self.uz = cost*np.sign(uz)	
            # SIGN() is faster than division.
        else:       # regular incident.
            temp = (1.0 - uz*uz)**0.5
            self.ux = sint*(ux*uz*cosp - uy*sinp)/temp + ux*cost
            self.uy = sint*(uy*uz*cosp + ux*sinp)/temp + uy*cost
            self.uz = -sint*cosp*temp + uz*cost

        if verbose: print('Spin')

    def record_R(self, model, refl):
        # Record the photon weight exiting the first layer(uz<0), 
        # no matter whether the layer is glass or not, to the 
        # reflection array.
        # Update the photon weight as well.
        x = self.x
        y = self.y
  
        ir = int((x*x + y*y)**0.5/model.dr)
        if ir > (model.nr - 1):
            ir = (model.nr - 1)
  
        ia = int(np.arccos(-self.uz)/model.da) 
        if ia > (model.na - 1):
            ia = model.na - 1
  
        # assign photon to the reflection array element.
        model.Rd_ra[ir, ia] += self.w*(1.0 - refl)
        self.w *= refl

    def record_T(self, model, refl):
        # Record the photon weight exiting the last layer(uz>0), 
        # no matter whether the layer is glass or not, to the 
        # transmittance array.
        # Update the photon weight as well.
  
        x = self.x
        y = self.y
 
        ir = int((x*x + y*y)**0.5/model.dr)
        if ir > (model.nr - 1):
            ir = model.nr - 1
  
        ia = int(np.arccos(self.uz)/model.da)
        if ia > (model.na - 1):
            ia = model.na - 1
  
        # assign photon to the transmittance array element.
        model.Tt_ra[ir, ia] += self.w*(1.0 - refl) 
        self.w *= refl

    def roulette(self):
        # The photon weight is small, and the photon packet tries 
        # to survive a roulette.
        if self.w == 0.0:	
            self.dead = True
        elif np.random.random_sample() < CHANCE:    # survived the roulette.
            self.w /= CHANCE
        else: 
            self.dead = True

    def tiltRotate(self,u,v,w,cost):
        while True :  # generate random azimuthal angle
            x1=np.random.random_sample()*2-1  # cos(phi); phi=azimuthal
            x2=np.random.random_sample()*2-1  # sin(phi); phi=azimuthal
            x3=x1*x1+x2*x2           # uniform distribution between [0,1]
            if (x3<=1) : break  # reject pair if not in unit circle; still faster than trig fcn

        # Jacques2011 Eq5.47a: sin(theta)=sqrt(1-cost*cost))
        if (abs(w) < 0.9) :  # tilt by polar angle and rotate by azimuthal angle
            uu = cost * u + ((1-cost*cost)/(1-w*w)/x3)**0.5 * (x1*u*w - x2*v)
            v  = cost * v + ((1-cost*cost)/(1-w*w)/x3)**0.5 * (x1*v*w + x2*u)
            w  = cost * w - ((1-cost*cost)*(1-w*w)/x3)**0.5 * x1
        else :
            uu = cost * u + ((1-cost*cost)/(1-v*v)/x3)**0.5 * (x1*u*v + x2*w)
            w  = cost * w + ((1-cost*cost)/(1-v*v)/x3)**0.5 * (x1*v*w - x2*u)
            v  = cost * v - ((1-cost*cost)*(1-v*v)/x3)**0.5 * x1
        u = uu

        return u,v,w
    
    def startLambertian(self, u,v,w): # Initiate Lambertian oriented in [u,v,w] direction
        cost = np.random.random_sample()**0.5  # cos(theta)

        [u,v,w] = self.tiltRotate(u,v,w,cost)
        return u,v,w

    def startGaussian(self, u,v,w, GaussCDF, cdfResolution):
        rand = np.random.random_sample()  # Random number in range [0,1]
        found = False; i = 0
        while ~found:
            if GaussCDF[i]<rand:  # keep going
                i = i + 1
            else:
                break  # leave loop when index is found near cdf=rand
        th = i/cdfResolution * np.pi/2  # random generated polar angle following Gaussian distribution      
        cost = np.cos(th)  # cos(theta)

        [u,v,w] = self.tiltRotate(u,v,w,cost)
        return u,v,w
    
    def startMeasBeam(self, u,v,w, MeasBeamCDF, cdfResolution):
        rand = np.random.random_sample()  # Random number in range [0,1]
        found = False; i = 0
        while ~found:
            if MeasBeamCDF[i]<rand:  # keep going
                i = i + 1
            else:
                break  # leave loop when index is found near cdf=rand
        th = i/(cdfResolution+1) * np.pi/2  # random generated polar angle following Gaussian distribution
        cost = np.cos(th)  # cos(theta)

        [u,v,w] = self.tiltRotate(u,v,w,cost)
        return u,v,w

    def RFresnel(self, n1, n2, ca1):
        # Compute the Fresnel reflectance.
        # Make sure that the cosine of the incident angle a1
        # is positive, and the case when the angle is greater 
        # than the critical angle is ruled out.
        # Avoid trigonometric function operations as much as
        # possible, because they are computation-intensive.
    
        if n1 == n2:			  	# matched boundary
            ca2 = ca1
            r = 0.0
        elif ca1 > COSZERO:     # normal incident
            ca2 = ca1
            r = (n2-n1)/(n2+n1)
            r *= r
        elif ca1 < COS90D:      # very slant
            ca2 = 0.0
            r = 1.0
        else:           # general	
            # sine of the incident and transmission angles
            sa1 = (1.0 - ca1*ca1)**0.5
            sa2 = n1*sa1/n2
            if sa2 >= 1.0:
                # double check for total internal reflection
                ca2 = 0.0
                r = 1.0
            else:
                # cosines of the sum ap or
                # difference am of the two
                # angles. ap = a1+a2
                # am = a1 - a2     
                ca2 = (1.0 - sa2*sa2)**0.5;     
                cap = ca1*ca2 - sa1*sa2     # c+ = cc - ss
                cam = ca1*ca2 + sa1*sa2     # c- = cc + ss
                sap = sa1*ca2 + ca1*sa2     # s+ = sc + cs
                sam = sa1*ca2 - ca1*sa2     # s- = sc - cs
                r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam) 
        return r, ca2


    def SpinTheta(self, g):
        # Choose (sample) a new theta angle for photon propagation
        # according to the anisotropy.
        # If anisotropy g is 0, then
        # cos(theta) = 2*rand-1.
        # otherwise
        # sample according to the Henyey-Greenstein function.
        # Returns the cosine of the polar deflection angle theta.

        if g == 0.0: 
            cost = 2*np.random.random_sample() - 1
        else:
            temp = (1 - g*g)/(1 - g + 2*g*np.random.random_sample())
            cost = (1 + g*g - temp*temp)/(2*g)
            if cost < -1:
                cost = -1.0
            elif cost > 1:
                cost = 1.0
        return cost




