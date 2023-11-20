import configparser glob, json
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord,match_coordinates_sky

# Read in the configuration file
cfg = configparser.ConfigParser()
cfg.read('config.ini')

sourceCoords = SkyCoord(cfg['POSITIONS']['source_ra'],cfg['POSITIONS']['source_dec'], frame='icrs')
flux_threshold   = float(cfg['THRESHOLDS']['flux_threshold'])
size_threshold   = float(cfg['THRESHOLDS']['size_threshold'])
target_threshold = float(cfg['THRESHOLDS']['target_threshold'])

# Load date-time array
date_times = np.load('../files/date-times.npy')
image_names = np.load('../files/image-names.npy')
beams   = []
beamextents = []

# Iterate through the files -- separating target and field sources
for image_name,date_time in zip(image_names[:],date_times[:]):
    # Extract the beam size
    hdu = fits.open(image_name)
    header = hdu[0].header
    bmaj = header['BMAJ'] 
    bmin = header['BMIN']
    bpa  = np.radians(header['BPA'])

    # Angular extents along x (RA) and y (Dec) directions equivalent to the FWHM projected on x and y plane
    dec_beam = 2.0 * np.sqrt((bmaj * 0.5) ** 2 * np.cos(bpa) ** 2  + (bmin * 0.5) ** 2 * np.sin(bpa) ** 2)
    ra_beam  = 2.0 * np.sqrt((bmaj * 0.5) ** 2 * np.sin(bpa) ** 2  + (bmin * 0.5) ** 2 * np.cos(bpa) ** 2)
    beamextents.append([ra_beam * 3600. ,dec_beam * 3600.]) # For saving a text file
    beams.append([bmaj * 3600.0, bmin * 3600., bpa]) # For saving a text file
    
    print('Filtering: ',date_time)
    fname = glob.glob('../files/total_field*{}*.fits'.format(date_time))[0] #get filename
    im = fits.open(fname)
    target = {'ra' : [], 'dec': [], 'peak':[], 'rms':[], 'ra_beam':[], 'dec_beam':[]}
    field  = {'ra' : [], 'dec': [], 'peak':[], 'rms':[], 'ra_beam':[], 'dec_beam':[]}

    # Extract data from FITS files
    ras         = im[1].data['RA']
    decs        = im[1].data['DEC']
    source_flux = im[1].data['Total_flux']     # Flux of the source (Jy)
    rms_err     = im[1].data['isl_rms']        # Get RMS value from background RMS-map local to the source
    peak_flux   = im[1].data['Peak_flux']      # Flux Density of the peak (Jy/beam)
    island_flux = im[1].data['Isl_Total_flux'] # Total Flux in the island (Jy)
    major       = im[1].data['Maj']            # Major Axis of the Source (deg)
    minor       = im[1].data['Min']            # Minor Axis of the Source (deg)
    stypes      = im[1].data['S_Code']
    ra_beams    = np.ones(len(im[1].data['RA'])) * ra_beam
    dec_beams   = np.ones(len(im[1].data['RA'])) * dec_beam

    # Make SkyCoord object from coordinates
    allCoords = SkyCoord(ra = ras * u.deg, dec = decs * u.deg, frame='icrs')
    separations = allCoords.separation(sourceCoords)

    # Find what indexes are target vs.field sources indexes
    target_index = np.where(separations.arcsec < target_threshold) # Any source within a beam major axis of the source coordinates is recorded as the target

    #######################################################################
    # Find the Field sources, enforcing point-source conditions           #
    #  Condition 1: (Default) Peak flux within 25% of the  island_flux    #
    #  Condition 2: (Default) Source shape within 25% of the beam shape   #
    #  Condition 3: It is not the target source                           #
    #######################################################################
    
    field_index = np.where((abs(peak_flux/island_flux - 1.0) < flux_threshold) & (abs(major/bmaj - 1.0) < size_threshold) & (abs(minor/bmin - 1.0) < size_threshold))[0] 
    field_index = np.setdiff1d(field_index,target_index) # Remove target from the field index

    # Fill target dictionary
    target['ra'], target['dec'], target['peak'], target['rms'], target['ra_beam'], target['dec_beam'] = ras[target_index].tolist(), decs[target_index].tolist(), peak_flux[target_index].tolist(), rms_err[target_index].tolist(), ra_beams[target_index].tolist(), dec_beams[target_index].tolist()

    # Fill field source dictionary
    field['ra'], field['dec'], field['peak'], field['rms'], field['ra_beam'], field['dec_beam'] = ras[field_index].tolist(), decs[field_index].tolist(), peak_flux[field_index].tolist(), rms_err[field_index].tolist(), ra_beams[field_index].tolist(), dec_beams[field_index].tolist()

    # Save the individual epoch dictionaries
    with open('../files/fieldsources_{}_{}.json'.format(date_time,cfg['SOURCE']['name']),'w') as jfile:
        json.dump(field,jfile)

    with open('../files/target_{}_{}.json'.format(date_time,cfg['SOURCE']['name']),'w') as jfile:
        json.dump(target,jfile)

#Save beamsizes as a text file
with open('../results/beams.txt', 'w') as tfile:
    np.savetxt(tfile,beams,header='BMAJ (arcsec), BMIN (arcsec), BPA (deg)')

#Save beamsizes as a text file
with open('../results/beamextents.txt', 'w') as tfile:
    np.savetxt(tfile,beamextents, header='Beam extent in right acension (arcsecond), Beam extent in declination (arcsecond)')
    
print('\n')
