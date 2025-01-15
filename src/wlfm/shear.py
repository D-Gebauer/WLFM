import numpy as np
import healpy as hp


class CreateNoiseReals(object):
    '''
    Class for generating noise fields.
    It is optimized to generate many realizations from
    a single galaxy shape catalog
    '''

    def __init__(self, ra, dec, e1, e2, w = None, NSIDE = 512):
        

        self.NSIDE = NSIDE
        self.pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)
        
        self.unique_pix, self.idx_rep = np.unique(self.pix, return_inverse=True)
        del ra, dec
        
        self.n_map = np.zeros(hp.nside2npix(self.NSIDE))
        self.n_map[self.unique_pix] += np.bincount(self.idx_rep, weights = w)
        
        if w is None: w = 1
        
        self.w = w
        
        #Only select pixels where we have at least a single galaxy
        #Rest will have zero ellipticity by default
        self.mask_sims = self.n_map != 0.

        self.e1 = e1
        self.e2 = e2


    def process(self, seed):
        
        e1_map    = np.zeros(hp.nside2npix(self.NSIDE))
        e2_map    = np.zeros(hp.nside2npix(self.NSIDE))
        
        rot_angle = np.random.default_rng(seed).random(self.e1.size)*2*np.pi
        e1, e2    = self.rotate_ellipticities(self.e1, self.e2, rot_angle)
        
        #Math for getting the weighted shape average per pixel
        e1_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e1 * self.w)
        e2_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e2 * self.w)
        e1_map[self.mask_sims]   = e1_map[self.mask_sims]/(self.n_map[self.mask_sims])
        e2_map[self.mask_sims]   = e2_map[self.mask_sims]/(self.n_map[self.mask_sims])
        
        return e1_map, e2_map
    

    def rotate_ellipticities(self, e1, e2, rot_angle):
        """
        Random rotate ellipticities e1 and e2 over
        angles given in `rot_angle`, which is in
        units of radians
        """
        #Rotate galaxy shapes randomly
        cos = np.cos(rot_angle)
        sin = np.sin(rot_angle)
        e1_new = + e1 * cos + e2 * sin
        e2_new = - e1 * sin + e2 * cos
        return e1_new, e2_new
        
        
class MakeMapFromCat(object):
    '''
    Class for generating shear map from  source galaxy catalogs.
    '''

    def __init__(self, ra, dec, e1, e2, w = None, NSIDE = 512):
        

        self.NSIDE = NSIDE
        self.pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)
        
        self.unique_pix, self.idx_rep = np.unique(self.pix, return_inverse=True)
        del ra, dec
        
        self.n_map = np.zeros(hp.nside2npix(self.NSIDE))
        self.n_map[self.unique_pix] += np.bincount(self.idx_rep, weights = w)
        
        if w is None: w = 1
        
        self.w = w
        
        #Only select pixels where we have at least a single galaxy
        #Rest will have zero ellipticity by default
        self.mask_sims = self.n_map != 0.

        self.e1 = e1
        self.e2 = e2


    def process(self):
        e1_map    = np.zeros(hp.nside2npix(self.NSIDE))
        e2_map    = np.zeros(hp.nside2npix(self.NSIDE))
        w_map     = np.zeros(hp.nside2npix(self.NSIDE))
        
        e1 = self.e1
        e2 = self.e2
        
        #Math for getting the weighted shape average per pixel
        e1_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e1*self.w)
        e1_map[self.unique_pix] /= self.n_map[self.unique_pix]
        e2_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e2*self.w)
        e2_map[self.unique_pix] /= self.n_map[self.unique_pix]
        w_map[self.unique_pix]  += np.bincount(self.idx_rep, weights = self.w)
        
        return e1_map, e2_map, w_map
    
    
    def combine_bins(map1, map2, w1, w2):
        """
        Combine two shear maps with weights
        """
        
        return (map1 * w1 + map2 * w2) / (w1 + w2), w1 + w2
    
    
def rotate_map(maps, rot_angles, nside, flip=False):

    alpha, delta = hp.pix2ang(nside, np.arange(maps.shape[1]))
    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if flip:
        rot_i = hp.ang2pix(nside, np.pi - rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(nside, rot_alpha,         rot_delta)
    rot_map = np.zeros_like(maps)
    rot_map[:, rot_i] =  maps[:, np.arange(maps.shape[1])]
    return rot_map


def rotate_like_des(maps, rot_number):
    '''
    rot_number goes from 0 to 3
    rot_number goes from 0 to 3
    "maps" can be a (Nmaps, Npixel) array
    '''
    
    NSIDE = hp.npix2nside(maps.shape[1])

    if rot_number == 0:
        pass
    elif rot_number == 1:
        maps = rotate_map(maps, [180, 0, 0], NSIDE, flip=False)
    elif rot_number == 2:
        maps = rotate_map(maps, [90, 0, 0],  NSIDE, flip=True)
    elif rot_number == 3:
        maps = rotate_map(maps, [270, 0, 0], NSIDE, flip=True)

    return maps