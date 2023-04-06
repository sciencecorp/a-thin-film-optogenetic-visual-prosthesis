import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

def get_point_irradiance(x_test,y_test,z_test,power_uW,ledx, ledy, model, rn, zn, LEDarea_cm2):

    #power uW emission from LED
    power_W = power_uW / 1e6  # p to watts
    LEAarea_m2 = LEDarea_cm2 / 10000
    led_irradiance = power_W / LEAarea_m2 
    
    dist = np.linalg.norm(np.array((ledx,ledy)) - np.array((x_test,y_test)))

    if dist > 100:
        return 0
    
    #detector_size_meters = detector_size_um / 1e6
    loc_um = [dist,z_test]  #(r,z) coordinates of cell location, where LED is centered at (0,0)
    irnew = min(range(len(rn)), key=lambda i: abs(rn[i]-loc_um[0]))
    iznew = min(range(len(zn)), key=lambda i: abs(zn[i]-loc_um[1]))
    
    efficiency = model[irnew,iznew]
    
    W_um2 = (led_irradiance * efficiency) #/ (detector_size_meters * detector_size_meters)
    mW_um2 = W_um2 * 1000
    mW_cm2 = mW_um2 / 10000
    
    return mW_cm2

def get_spike_prob(model, irradiance, exposure_ms=5):
    return model(exposure_ms, irradiance)

def avg_nearby_voxels(array3d,centervalue,avgwidth):
    dum = array3d
    dum = dum[centervalue-avgwidth:centervalue+avgwidth+1,centervalue-avgwidth:centervalue+avgwidth+1]
    dum.shape  # to check size of multidimensional array
    dum = np.mean(dum,axis=(0,1))
    
    return dum

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

def get_mean_soma_irradiance_map(emission_profile, soma_diameter_um, LEDarea=None, r_aperture=None):

    # LED area cm2
    # r_aperature cm

    if r_aperture is None:
        r_aperture = 0   # cm
    if LEDarea is None:
        LEDarea = 0.0015**2  # cm2
    
    #if emission_profile not in ['lambertian', 'guassian', 'collimated', 'empirical', 'hinge', 'uturn']:
    #    raise Exception('emission_profile must be lambertian, guassian, empirical or collimated')

    if emission_profile == 'lambertian':
        loadpath = 'mcml_scattering_nph1e7_glass2retina_rAperture7um_lambertian.npy'
        [numPhotons,modelName,srcShape,md] = np.load(loadpath, allow_pickle=True)  # load results

    elif emission_profile == 'guassian':
        loadpath = 'mcml_scattering_nph1e6_glass2retina_rAperture15um_gaussian_HA8deg.npy'
        [numPhotons,modelName,srcShape,md] = np.load(loadpath, allow_pickle=True)  # load results

    elif  emission_profile == 'collimated':
        [numPhotons, b, c, d, md, f, g]=np.load('Collimated_nph1e6_glass2retina_rAperture15um.npy', allow_pickle=True)

    elif  emission_profile == 'hinge':
        [numPhotons, b, c, d, md, f, g]=np.load('/Users/alanmardinly/Documents/MeasBeam_6by11um_nph1e7_glass2retina_noAperture.npy', allow_pickle=True)
    
    elif  emission_profile == 'uturn1':
        [numPhotons, b, c, d, md, f, g]=np.load('/Users/alanmardinly/Documents/MeasBeam_15by19um_nph1e7_glass2retina_noAperture.npy', allow_pickle=True)
    
    elif  emission_profile == 'uturn2':
        [numPhotons, b, c, d, md, f, g]=np.load('/Users/alanmardinly/Documents/MeasBeam_20by34um_nph1e6_glass2retina_noAperture.npy', allow_pickle=True)


    dz = md.dz
    nz = md.nz
    dr = md.dr
    nr = md.nr
    ixy_cent = md.ixy_cent
    # average over center values to denoise

    #crop out first 10um of stack-up, to zero z at surface of passivation
    dz_crop = 10*1e-4  # cm
    iz_crop = int(dz_crop/dz)
    Flux_crop = md.Flux_xyz[:,:,iz_crop:]  #cropped flux

    flux_sc = Flux_crop/numPhotons*LEDarea/dr**2  # flux scaled to LEDirradiance=1
    rsoma = (soma_diameter_um / 2) / 10000  #convert from diameter um to radius cm

    irrOnSoma = np.zeros([np.shape(flux_sc)[0],np.shape(flux_sc)[2]])
    for iz in np.arange(np.shape(flux_sc)[2]):
        for ix in np.arange(np.shape(flux_sc)[0]):
            dat = flux_sc[:,:,iz]
            irrOnSoma[ix,iz] = avg_over_soma(dat,ix,ixy_cent,rsoma,dr)

    raxis = np.linspace(-dr*ixy_cent,dr*ixy_cent,np.shape(irrOnSoma)[0])*1e4
    zaxis = np.arange(0,dz*(nz-iz_crop),dz)*1e4


    from scipy import interpolate
    f = interpolate.interp2d(zaxis, raxis, irrOnSoma, kind='cubic')

    rnew = np.arange(-dr*ixy_cent,dr*(ixy_cent+1e-6),dr/10)*1e4  # um
    znew = np.arange(0,dz*(nz-iz_crop),dz/10)*1e4  # um
    irrOnSoma_new = f(znew,rnew)

    return irrOnSoma_new, rnew, znew

    #    # To find intensity at [r,z] coordinates
    #loc_um = [0,20]     #um; (r,z) coordinates of cell location, where device passivation is at (0,0)
    #irnew = min(range(len(rnew)), key=lambda i: abs(rnew[i]-loc_um[0]))
    #iznew = min(range(len(znew)), key=lambda i: abs(znew[i]-loc_um[1]))

    #print('Average LED irradiation received by a 15um diameter soma at (r,z)=(%.1f,%.1f)um  is:  %.3f' %(loc_um[0],loc_um[1],irrOnSoma_new[irnew,iznew]))


def get_led_locations(array):


    if array == 'hinge':
        nx = 128
        ny = 128
        pitch = 20
        k1 = 16
    elif array == 'uturn':
        nx = 64
        ny = 64
        pitch = 42
        k1 = 8


    LED_locations = np.empty((nx*ny,2))

    x_start = -((nx * pitch) / 2)
    y_start = -((ny * pitch) / 2)
    x = x_start
    y = y_start
    
    i = 0
    
    for k in range(nx):
        x += pitch
        y = y_start
        
        for j in range(ny):
            y += pitch
    
            LED_locations[i,0] = x
            LED_locations[i,1] = y
    
            i += 1

    # account for flexLED routing 
    revised_locations = []
    keep = True
    j = 0
    k = 0
    l = 0
    for i in range(len(LED_locations)):
        
        #if np.mod(i, 16) == 0:
        #    if keep == True:
        #        keep = False
        #    elif keep == False:
        #        keep = True
        if k == k1:
            k = 0
            if keep == True:
                keep = False
            elif keep == False:
                keep = True
        if l == nx:
            l = 0
            if keep == True:
                keep = False
            elif keep == False:
                keep = True
        l +=1
        k +=1
        
        
        if keep == True:
            revised_locations.append(LED_locations[i,:])
            j+=1

    revised_locations = np.array(revised_locations)
    
    return revised_locations


def simulate_irradiance_across_array(power_uw, rgc_xvals, rgc_yvals, z_sep, model, zn, rn, led_locations, LEDarea):

    results = np.empty((len(led_locations), len(rgc_xvals)))
  
    for led in range(len(led_locations)):  
        for rgc in range(len(rgc_xvals)): # get value of a single cell
            results[led, rgc]=get_point_irradiance(rgc_xvals[rgc],rgc_yvals[rgc],z_test=z_sep, power_uW=power_uw, ledx=led_locations[led, 0],ledy=led_locations[led, 1], model=model, rn=rn, zn=zn, LEDarea_cm2=LEDarea)

    return results

def plot_results(result):
    plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    
    plt.scatter(rgc_xvals,rgc_yvals, c=np.sum(result,0))
    ax.set_facecolor('tab:gray')

    plt.clim(0,20)
    plt.colorbar()

    plt.axis('equal')
    plt.show()


def get_num_addressable_cells(data, exposure_durations, spi):
    max_irradiance_by_cell = np.max(data['results'],0)
    number_of_addressable_cells = []
    for exposure_duration in exposure_durations:
        spike_prob = [get_spike_prob(spi, x, exposure_duration) for x in max_irradiance_by_cell]
        number_of_addressable_cells.append(np.sum(np.array(spike_prob)>.9))

    return number_of_addressable_cells

def get_num_useful_pixels(data, exposure_durations, spi):
    # how many pixels can we address at least one cell by exposure duration?
    number_of_useful_pixels = []
    max_irradiance_by_pixel = np.max(data['results'],1)
    npix = len(max_irradiance_by_pixel)
    for exposure_duration in exposure_durations:
        spike_prob = [get_spike_prob(spi, x, exposure_duration) for x in max_irradiance_by_pixel]
        number_of_useful_pixels.append(np.sum(np.array(spike_prob)>.9) / npix)

    return number_of_useful_pixels

def get_total_number_of_activatable_cells(data, exposure_durations, spi):

        # how many cells can we address total via the whole array by exposure duration?
    sum_irradiance_by_cell = np.sum(data['results'],0)
    number_of_total_addressable_cells = []
    for exposure_duration in exposure_durations:
        spike_prob = [get_spike_prob(spi, x, exposure_duration) for x in sum_irradiance_by_cell]
        number_of_total_addressable_cells.append(np.sum(np.array(spike_prob)>.9)) 

    return number_of_total_addressable_cells

def get_mean_selectivity(data, exposure_durations, spi):


    #spi is spike probability interpolation model

    activation_profile =[]
    for exposure_duration in exposure_durations:
        selective_activations = []

        for irradiance in range(1,22):
            if get_spike_prob(spi, irradiance, exposure_duration) > 0.9:
                threshold = irradiance
                break
        for cell in range(np.shape(data['results'])[1]):
            selective_activations.append(np.sum(data['results'][:,cell] > threshold))
        selective_activations = np.array(selective_activations)
        activation_profile.append(np.mean(selective_activations[selective_activations>0]))

    return activation_profile

def cell_activations_per_pixel(data, exposure_durations, spi):


    #spi is spike probability interpolation model

    activation_profile =[]
    for exposure_duration in exposure_durations:
        activation_per_pixel = []

        for irradiance in range(1,22):
            if rgcm.get_spike_prob(spi, irradiance, exposure_duration) > 0.9:
                threshold = irradiance
                break
        for pixel in range(np.shape(data['results'])[0]):
            activation_per_pixel.append(np.sum(data['results'][pixel,:] > threshold))
        selective_activations = np.array(activation_per_pixel)
        activation_profile.append(np.mean(selective_activations[selective_activations>0]))

    return activation_profile