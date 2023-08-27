import configparser, glob
from astropy.io import fits
import bdsf #PyBDSF
import numpy as np

# Read in the configuration file
cfg = configparser.ConfigParser()
cfg.read('config.ini')

date_times = []

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))


fix_to_beam = str_to_bool(cfg['PYBDSF']['fix_to_beam'])

# Extract the source name, observing times, and order according to date
image_names = glob.glob('../images/*.fits') #get image names
for image_name in image_names:
    hdul = fits.open(image_name) #load image
    date_times.append(hdul[0].header['DATE-OBS'])
    hdul.close()

# Ensure it's ordered in time
sorted_index = sorted(range(len(date_times)), key = date_times.__getitem__)
date_times = np.array(date_times)[sorted_index]
image_names = np.array(image_names)[sorted_index]

# Save the date-times for future scripts
np.save('../files/date-times', date_times)
np.save('../files/image-names',image_names)

'''
# Run each image through PyBDSF
for image_name,date_time in zip(image_names,date_times): #All images

    # PyBDSF run
    img = bdsf.process_image(image_name,thresh_pix=float(cfg['PYBDSF']['snr_threshold']),fix_to_beam=fix_to_beam, minpix_isl=5.0,  rms_box=(300,100)) #trim_box=(7750,8250,7750,8250))

    # Save FITS catalog
    img.write_catalog(format='fits', catalog_type='srl', outfile = '../files/total_field_%s_%s.fits' %(date_time,cfg['SOURCE']['name']), clobber=True)
'''
