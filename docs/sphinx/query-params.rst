
.. _marvin-query-parameters:

Query Parameters
================

This is a list of available parameters returnable in your Query that have been vetted and verified.  The naming conventions here are the same for the filter parameter names.  There are more parameters.  Please let us know which ones you wish to be made available, and we will add them.

General Cube/Map Properties
---------------------------
* **cube.plateifu**: The plate+ifudesign name for this object
* **cube.mangaid**: The mangaid for this object
* **cube.ra**: OBJRA - Right ascension of the science object in J2000
* **cube.dec**: OBJDEC - Declination of the science object in J2000
* **cube.plate**: The plateid
* **bintype.name**: The type of binning used in DAP maps
* **template.name**: The stellar libary template used in DAP maps

Spaxel Properties
-----------------
* **spaxelprop.x**: The spaxel x position
* **spaxelprop.y**: The spaxel y position
* **spaxelprop.emline_gflux_ha_6564**: Gaussian profile integrated flux for Ha emission line
* **spaxelprop.emline_gflux_hb_4862**: Gaussian profile integrated flux for Hb emission line
* **spaxelprop.emline_gflux_nii_6549**: Gaussian profile integrated flux for NII emission line
* **spaxelprop.emline_gflux_nii_6585**: Gaussian profile integrated flux for NII emission line
* **spaxelprop.emline_gflux_oiid_3728**: Gaussian profile integrated flux for OIId emission line
* **spaxelprop.emline_gflux_oiii_4960**: Gaussian profile integrated flux for OIII emission line
* **spaxelprop.emline_gflux_oiii_5008**: Gaussian profile integrated flux for OIII emission line
* **spaxelprop.emline_gflux_sii_6718**: Gaussian profile integrated flux for SII emission line
* **spaxelprop.emline_gflux_sii_6732**: Gaussian profile integrated flux for SII emission line
* **spaxelprop.nii_to_ha**: The NII/Ha ratio computed from emline_gflux
* **spaxelprop.oiii_to_hb**: The OIII/Hb ratio computed from emline_gflux
* **spaxelprop.sii_to_ha**: The SII/Ha ratio computed from emline_gflux
* **spaxelprop.ha_to_hb**: The Ha/Hb ratio computed from emline_gflux
* **spaxelprop.emline_gvel_ha_6564**: Gaussian profile velocity for Ha emission line
* **spaxelprop.emline_gvel_oiii_5008**: Gaussian profile velocity for OIII emission line
* **spaxelprop.emline_gsigma_ha_6564**: Gaussian profile velocity dispersion for Ha emission line; must be corrected using EMLINE_INSTSIGMA
* **spaxelprop.emline_gsigma_oiii_5008**: Gaussian profile velocity dispersion for OIII emission line; must be corrected using EMLINE_INSTSIGMA
* **spaxelprop.stellar_vel**: Stellar velocity relative to NSA redshift
* **spaxelprop.stellar_sigma**: Stellar velocity dispersion (must be corrected using STELLAR_SIGMACORR)
* **spaxelprop.specindex_d4000**: Measurements of spectral indices

NSA Properties
--------------
* **nsa.iauname**: The accepted IAU name
* **nsa.ra**: Right ascension of the galaxy
* **nsa.dec**: Declination of the galaxy
* **nsa.z**: The heliocentric redshift
* **nsa.elpetro_ba**: Axis ratio b/a from elliptical petrosian fit.
* **nsa.elpetro_mag_g_r**: g-r color computed from the Azimuthally-averaged SDSS-style Petrosian flux in FNugriz
* **nsa.elpetro_logmass**: Log of the stellar mass from K-correction fit in h-2 solar masses to elliptical petrosian magnitudes.
* **nsa.elpetro_th50_r**: Elliptical petrosian 50% light radius (derived from r band), in arcsec.
* **nsa.sersic_logmass**: Log of the stellar mass from 2D Sersic fit
* **nsa.sersic_ba**: Axis ratio b/a from 2D Sersic fit.

