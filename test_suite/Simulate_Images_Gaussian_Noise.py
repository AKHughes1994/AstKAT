from utils import *
#np.random.seed(66642070) 

def RootMeanSqr(data):
    return np.sqrt(np.mean(data ** 2))

def RandomizePSF():
    
    # Fix PSF minor axis as 5 pixels
    bmin = 5.0
       
    # Allow major axis to be up to 3x the size of the minor and BPA to be random
    bmaj = bmin + 10.0 * np.random.rand(1)
    bpa  = np.pi * np.random.rand(1) - np.pi * 0.5

    # Return PSF parameters
    return {'bmaj': bmaj[0], 
            'bmin': bmin , 
            'bpa': bpa[0], 
            'sigma_maj': bmaj[0] / (8 * np.log(2)) ** 0.5, 
            'sigma_min': bmin / (8 * np.log(2)) ** 0.5}    

def InitializeHeader(PSF, iteration = 0):

    header = fits.Header()

    # Based on WSCLEAN output header for L band data
    header['simple']   = 'T'
    header['bitpix']   = -32
    header['NAXIS1']   = imsize
    header['NAXIS2']   = imsize
    header['NAXIS3']   = 1
    header['NAXIS4']   = 1
    header['BSCALE']   = 1.0e0
    header['BZERO']    = 0.0e0
    header['BUNIT']    = 'Jy/beam'
    header['BMAJ']     = PSF['bmaj'] / 3600.0
    header['BMIN']     = PSF['bmin'] / 3600.0
    header['BPA']      = np.degrees(PSF['bpa'])
    header['EQUINOX']  = 2000
    header['BTYPE']    = 'Intensity'

    header['CTYPE1']   = 'RA---SIN'
    header['CRPIX1']   = int(imsize * 0.5 + 1)
    header['CRVAL1']   = -8.731470833333E+01
    header['CDELT1']   = -2.777777777778E-04
    header['CUNIT1']   = 'deg'

    header['CTYPE2']   = 'DEC--SIN'
    header['CRPIX2']   = int(imsize * 0.5 + 1)
    header['CRVAL2']   = 0.0E1
    header['CDELT2']   = 2.777777777778E-04
    header['CUNIT2']   = 'deg'

    header['CTYPE3'] = 'FREQ'
    header['CRPIX3'] = 1
    header['CRVAL3'] = 1.283986938477E+09
    header['CDELT3'] = 8.560000000000E+08
    header['CUNIT3'] = 'Hz'

    header['CTYPE4'] = 'STOKES'
    header['CRPIX4'] = 1
    header['CRVAL4'] = 1.000000000000E+00
    header['CDELT4'] = 1.000000000000E+00
    header['CUNIT4'] = None

    header['SPECSYS']  = 'TOPOCENT'
    header['DATE-OBS'] = f'{iteration:04d}-09-20T00:00:00'

    return header


def GenerateRandomImage(iteration):

    # Generate a random PSF and initialize the simulated WSCLEAN header
    PSF = RandomizePSF()
    header = InitializeHeader(PSF, iteration = iteration)

    # Initialize the noise map and smooth according to PSF
    noise = np.random.randn(imsize, imsize) # Gaussian noise
    smoothing_kernel = Gaussian2DKernel(x_stddev = PSF['sigma_maj'], 
                                        y_stddev=PSF['sigma_min'], 
                                        theta = PSF['bpa'] + 0.5 * np.pi, 
                                        x_size=51 , 
                                        y_size=51, 
                                        mode='center').array
    noise = fftconvolve(noise, smoothing_kernel, mode='same') 
    noise /= RootMeanSqr(noise) # Re-normalize the noise so the RMS is 1
        
    # Add scatter according to assumed Rayleighian Astrometric error distibution
    snr = amplitude # RMS is 1
    theta_pa  = 2 * np.pi * np.random.rand(n_comps) # Random offset position angle
    sigma_hat = AstrometricError(snr, A, B) # Dimensionless error
    x_sep  = []
    y_sep  = []

    for k in range(n_comps):
        
        # Solve for Dr
        semi_major = 1.5096 * PSF['sigma_maj']
        semi_minor = 1.5096 * PSF['sigma_min']

        trans_pa = theta_pa[k] - PSF['bpa']
        x68 = (semi_major ** (-2) + np.tan(trans_pa) ** 2 * semi_minor ** (-2)) ** (-0.5)
        y68 = np.tan(trans_pa) * x68
        r68 = (x68 ** 2 + y68 ** 2) ** (0.5) 

        # Draw a random separation from a Rayleigh disribution with scale factor sigma_hat
        sep = r68 * rayleigh.rvs(scale = sigma_hat[k])

        # Decompose these into ra and dec offsets
        x_sep.append(-sep * np.cos(theta_pa[k] + np.pi/2)) # Convert to asec
        y_sep.append(+sep * np.sin(theta_pa[k] + np.pi/2))


    # Make a Table to store the PhotUtils Generated Guassian Components
    source_table = QTable()
    source_table['amplitude'] = amplitude
    source_table['x_mean'] = x_true + x_sep
    source_table['y_mean'] = y_true - y_sep
    source_table['x_stddev'] = [PSF['sigma_maj']] * n_comps
    source_table['y_stddev'] = [PSF['sigma_min']] * n_comps
    source_table['theta'] = [PSF['bpa'] + 0.5 * np.pi] * n_comps
    sources = make_gaussian_sources_image((imsize, imsize), source_table)

    # Sum noise + source emision and save as fits for processing
    image = noise + sources
    hdu = fits.PrimaryHDU(data=image[np.newaxis, np.newaxis, :, :], header = header)
    hdu.writeto(f'{images}/Simulated_Image_{iteration:04d}.fits', overwrite=True)

def main():

    # NOTE: Pixels are assumes to be 1 arcsec, I don't see a reason to make it more complicated than that

    # Initialize the global variables directory labels
    global images
    global imsize
    global n_image
    global n_comps
    global A
    global B
    global amplitude
    global x_true
    global y_true

    # Directory to save images
    images = os.getcwd().replace('scripts', 'images')
    
    # Feel free to modify the Global paramaters
    imsize  = 2000
    n_image = 100
    n_comps = 250
    A = 0.0 # Add a random B-term the A-term will be set by the Gaussian noise
    B = 0.037
    amplitude = truncnorm.rvs(scale = 30, 
                    loc=4.0, 
                    a = (4.0 - 4.0) / 30.0, 
                    b = (1000.0 - 4.0) / 30.0, 
                    size = n_comps)
    x_true = np.random.rand(n_comps) * (imsize - 30) + 15 # Avoid edges
    y_true = np.random.rand(n_comps) * (imsize - 30) + 15 

    # Make simulated images
    for k in range(n_image):
        msg(f'Simulating image {k:04d}')
        GenerateRandomImage(iteration = k)

    return 0

if __name__ in "__main__":
    main() 
