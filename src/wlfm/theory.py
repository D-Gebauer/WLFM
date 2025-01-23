import jax.numpy as jnp
from scipy import interpolate
import numpy as np
import healpy as hp
import cloudpickle
import tensorflow as tf

data_dir = "/home/moon/dgebauer/research/DLWL/data/emulators/"
train_dir = data_dir + "train/"
model_dir = '/home/moon/dgebauer/research/DLWL/models/emulators/'
conv_fac_dir = model_dir + "conversion_factors/"


param_names = ['Om', 'Ob', 'H0', 'ns', 'sigma8', 'w0', 'A_ia', 'c_min', 'dz1', 'dz2', 'dz3', 'dz4', 'm1', 'm2', 'm3', 'm4']

def get_random_params(n):
    Om_min, Om_max = 0.1, 0.5
    Ob_min, Ob_max = 0.02, 0.05
    H0_min, H0_max = 60, 75
    ns_min, ns_max = 0.8, 1.1
    sigma8_min, sigma8_max = 0.6, 1.0
    w0_min, w0_max = -2.0, -0.33
    A_ia_min, A_ia_max = -2.0, 2.0
    c_min_min, c_min_max = 0.5, 5.5
    dz1_min, dz1_max = -0.05, 0.05
    dz2_min, dz2_max = -0.05, 0.05
    dz3_min, dz3_max = -0.05, 0.05
    dz4_min, dz4_max = -0.05, 0.05
    m1_min, m1_max = -0.05, 0.05
    m2_min, m2_max = -0.05, 0.05
    m3_min, m3_max = -0.05, 0.05
    m4_min, m4_max = -0.05, 0.05

    min_list = [Om_min, Ob_min, H0_min, ns_min, sigma8_min, w0_min, A_ia_min, c_min_min, dz1_min, dz2_min, dz3_min, dz4_min, m1_min, m2_min, m3_min, m4_min]
    max_list = [Om_max, Ob_max, H0_max, ns_max, sigma8_max, w0_max, A_ia_max, c_min_max, dz1_max, dz2_max, dz3_max, dz4_max, m1_max, m2_max, m3_max, m4_max]

    return np.random.uniform(low=min_list, high=max_list, size=(n,16))

z_list_integ = jnp.linspace(0, 2.5, 250)
n_z_los = 250

ell = jnp.unique(jnp.logspace(np.log10(2), jnp.log10(15000), 88).astype(int))
fullell = jnp.linspace(2.0, 14998, 14997)
fullell_kitching = jnp.linspace(2.5, 14998.5, 14997)
n_ell = ell.size
n_fullell = fullell.size

pixwin = hp.pixwin(512, lmax=np.max(ell)) 
pixwin_ell = np.arange(len(pixwin))                                         # pixwin creates window function for ell=0 to 3*nside-1
pixwin = pixwin[np.intersect1d(ell, pixwin_ell, return_indices=True)[2]]    # match pixwin function to correct ells
pixwin = np.append(pixwin, np.zeros(len(ell) - len(pixwin)))                # Add zeros to match length
pixwin = pixwin.reshape((-1,n_ell)) # Expand the pixwin functions to all sampled nodes

n_angular_bins = 15
binedges = jnp.geomspace(10, 250, n_angular_bins+1)
r_min_xi = binedges[:-1]
r_max_xi = binedges[1:]
r_xi = jnp.sqrt(r_min_xi*r_max_xi)

n_s_z_BIN_z_tab = np.loadtxt('/home/moon/dgebauer/research/i3PCF/data/nofz/DESY3_nofz/nofz_DESY3_source_BIN1.tab', usecols=(0))
n_s_z_BIN1_vals = jnp.array(np.loadtxt('/home/moon/dgebauer/research/i3PCF/data/nofz/DESY3_nofz/nofz_DESY3_source_BIN1.tab', usecols=(1), unpack=True))
n_s_z_BIN1_vals /= jnp.trapezoid(n_s_z_BIN1_vals, n_s_z_BIN_z_tab)
n_s_z_BIN2_vals = jnp.array(np.loadtxt('/home/moon/dgebauer/research/i3PCF/data/nofz/DESY3_nofz/nofz_DESY3_source_BIN2.tab', usecols=(1), unpack=True))
n_s_z_BIN2_vals /= jnp.trapezoid(n_s_z_BIN2_vals, n_s_z_BIN_z_tab)
n_s_z_BIN3_vals = jnp.array(np.loadtxt('/home/moon/dgebauer/research/i3PCF/data/nofz/DESY3_nofz/nofz_DESY3_source_BIN3.tab', usecols=(1), unpack=True))
n_s_z_BIN3_vals /= jnp.trapezoid(n_s_z_BIN3_vals, n_s_z_BIN_z_tab)
n_s_z_BIN4_vals = jnp.array(np.loadtxt('/home/moon/dgebauer/research/i3PCF/data/nofz/DESY3_nofz/nofz_DESY3_source_BIN4.tab', usecols=(1), unpack=True))
n_s_z_BIN4_vals /= jnp.trapezoid(n_s_z_BIN4_vals, n_s_z_BIN_z_tab)
n_s_z_BIN1 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN1_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN2 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN2_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN3 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN3_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN4 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN4_vals, fill_value=(0,0), bounds_error=False)

FT_plus_binave_xi = jnp.load('/home/moon/ahalder/Integrated_3PCF/i3PCF/output/angular_bins/xip_10_250_15_bin_averaged_values.npy').T
FT_minus_binave_xi = jnp.load('/home/moon/ahalder/Integrated_3PCF/i3PCF/output/angular_bins/xim_10_250_15_bin_averaged_values.npy').T

P_shift = jnp.load(train_dir + "P_shift.npy")
H_chi_D_mean = jnp.load(conv_fac_dir + "H_chi_D_mean.npy")
H_chi_D_range = jnp.load(conv_fac_dir + "H_chi_D_range.npy")

emulator_H = tf.saved_model.load(model_dir + 'gpflow_H')
emulator_chi = tf.saved_model.load(model_dir + 'gpflow_chi')
emulator_D = tf.saved_model.load(model_dir + 'gpflow_D')

with open(model_dir + 'P_emulator.pkl', 'rb') as f:
    P_emulator = cloudpickle.load(f)

# Takes ['Omega_m', 'w0', 'z'] as input
def get_H_chi_D(theta):
    
    pred_H = jnp.asarray(emulator_H.predict_f_compiled(theta)[0].numpy())
    pred_chi = jnp.asarray(emulator_chi.predict_f_compiled(theta)[0].numpy())
    pred_D = jnp.asarray(emulator_D.predict_f_compiled(theta)[0].numpy())
    pred = jnp.concatenate([pred_H, pred_chi, pred_D], axis=1)
    print(pred.shape)
    return (pred * H_chi_D_range) + H_chi_D_mean

# Takes ['Omega_m', 'Omega_b', 'h', 'ns', 'sigma8', 'w0', 'z'] as input
def get_P_from_NN(theta):
    pred = jnp.pow(10, P_emulator(theta)) - P_shift
    return pred

#### Function computing H, chi and q from emulator ####
def get_H_chi_q_from_NN(theta):
    
    theta_H_chi_q = jnp.concat([jnp.reshape(theta[:,0],[-1,1]), jnp.reshape(theta[:,1],[-1,1])], axis=1)
    
    z_los_Wk = jnp.repeat(jnp.reshape(z_list_integ, [1,1,-1]), theta.shape[0], axis=0)
    z_los_Wk = jnp.repeat(z_los_Wk, n_z_los, axis=1)
    z_expansion = jnp.reshape(jnp.tile(z_list_integ, [theta.shape[0]]), [-1,1])
    theta_expansion = jnp.repeat(theta_H_chi_q, jnp.size(z_list_integ), axis=0)
    theta_expansion = jnp.concat([theta_expansion, z_expansion], axis=1)
    theta_expansion = jnp.astype(theta_expansion, jnp.float32)
    
    H, chi, D = get_H_chi_D(theta_expansion).T
    
    H = jnp.reshape(H, [theta.shape[0], n_z_los])
    H_0 = jnp.astype(jnp.repeat(jnp.reshape(theta[:,2], [-1,1]), n_z_los, axis=1), jnp.float32)
    H_0_cosmogrid = 67.3 #the fiducial current expansion rate of Cosmogrid
    H = H*H_0/H_0_cosmogrid
    
    chi = jnp.reshape(chi, [theta.shape[0], n_z_los])
    chi = chi*H_0_cosmogrid/H_0
    
    D = jnp.reshape(D, [theta.shape[0], n_z_los])

    H_0 = theta[:,2]/299792.458 
    Omega_m0 = theta[:,0]
    A_IA_0_NLA = theta[:,6]
    alpha_IA_0_NLA = jnp.zeros(theta.shape[0])
    
    H_0 = jnp.repeat(jnp.reshape(H_0, [-1,1]), n_z_los, axis=1)
    Omega_m0 = jnp.repeat(jnp.reshape(Omega_m0, [-1,1]), n_z_los, axis=1)
    A_IA_0_NLA = jnp.repeat(jnp.reshape(A_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    alpha_IA_0_NLA = jnp.repeat(jnp.reshape(alpha_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    
    # make two copies of chi as the chi_z and chi_zs components in the q kernel computation
    chi_z = jnp.copy(chi)
    chi_zs = jnp.copy(chi)
    
    chi_z = jnp.reshape(chi_z, [theta.shape[0], n_z_los, -1])
    chi_zs = jnp.reshape(chi_zs, [theta.shape[0], -1, n_z_los])
    chi_z = jnp.repeat(chi_z, n_z_los, axis=2)
    chi_zs = jnp.repeat(chi_zs, n_z_los, axis=1)
    
    theta_ns = theta
    n_zs_los_BIN1, n_zs_los_BIN2, n_zs_los_BIN3, n_zs_los_BIN4 = n_s_z_compute(theta_ns, z_list_integ, n_s_z_BIN1, n_s_z_BIN2, n_s_z_BIN3, n_s_z_BIN4)
    n_zs_los_BIN1_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN1, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN2_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN2, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN3_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN3, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN4_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN4, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    
    indicator = jnp.ones((n_z_los, n_z_los))
    indicator = jnp.triu(indicator, 0)
    indicator = jnp.repeat(jnp.reshape(indicator, [-1, n_z_los, n_z_los]), theta.shape[0], axis=0)
    
    W1_integrand = n_zs_los_BIN1_prime * (chi_zs - chi_z)/chi_zs * indicator
    W2_integrand = n_zs_los_BIN2_prime * (chi_zs - chi_z)/chi_zs * indicator
    W3_integrand = n_zs_los_BIN3_prime * (chi_zs - chi_z)/chi_zs * indicator
    W4_integrand = n_zs_los_BIN4_prime * (chi_zs - chi_z)/chi_zs * indicator
    W1 = jnp.trapezoid(W1_integrand, z_los_Wk, axis=2)
    W2 = jnp.trapezoid(W2_integrand, z_los_Wk, axis=2)
    W3 = jnp.trapezoid(W3_integrand, z_los_Wk, axis=2)
    W4 = jnp.trapezoid(W4_integrand, z_los_Wk, axis=2)
    
    f_IA_NLA_z = -A_IA_0_NLA * 0.0134 * Omega_m0 * ((1.0 + z_list_integ[None,:])/1.62)**alpha_IA_0_NLA/D
    
    q1 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W1 + f_IA_NLA_z * H * n_zs_los_BIN1
    q2 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W2 + f_IA_NLA_z * H * n_zs_los_BIN2
    q3 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W3 + f_IA_NLA_z * H * n_zs_los_BIN3
    q4 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W4 + f_IA_NLA_z * H * n_zs_los_BIN4
    
    return H, chi, q1, q2, q3, q4



#### A scipy function which compute n(z) w.r.t different photometric redshift uncertainty ####
def n_s_z_compute(theta, z_list_integ, n_s_z_BIN1, n_s_z_BIN2, n_s_z_BIN3, n_s_z_BIN4):
    
    z = jnp.repeat(z_list_integ.reshape(1,-1), theta.shape[0], axis=0)
    delta_z_1 = theta[:,8]
    delta_z_2 = theta[:,9]
    delta_z_3 = theta[:,10]
    delta_z_4 = theta[:,11]
    z_1 = z - delta_z_1[:,None]
    z_2 = z - delta_z_2[:,None]
    z_3 = z - delta_z_3[:,None]
    z_4 = z - delta_z_4[:,None]
    ns_z_BIN1_los = n_s_z_BIN1(z_1)
    ns_z_BIN2_los = n_s_z_BIN2(z_2)
    ns_z_BIN3_los = n_s_z_BIN3(z_3)
    ns_z_BIN4_los = n_s_z_BIN4(z_4)
    return ns_z_BIN1_los, ns_z_BIN2_los, ns_z_BIN3_los, ns_z_BIN4_los


#### A scipy function which interpolates the projected spectrum to all multipole numbers ###
def interpolation(ell, spectrum, fullell_kitching):
    interp = interpolate.interp1d(ell, spectrum, kind='cubic', axis=1, fill_value='extrapolate')
    spectrum_new = interp(fullell_kitching)
    return spectrum_new

#### Function carrying out kitching correction and converting the integrated bispectrum to correlation function ####
def kitching_correction_and_FT_plus_zeta(theta, spectrum, FT_plus_kernel):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_plus_kernel[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_minus_zeta(theta, spectrum, FT_minus_kernel):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_minus_kernel[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_plus_xi(theta, spectrum):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_plus_binave_xi[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_minus_xi(theta, spectrum):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_minus_binave_xi[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

#### LOS projection functions ####
def los_proj_power_spectrum(theta, q1, q2, q3, q4, chi, H, P):
    z_los = jnp.repeat(jnp.reshape(z_list_integ, [1, -1, 1]), ell.size, axis=2)
    z_los = jnp.repeat(z_los, theta.shape[0], axis=0)
    
    P_11_integ = (1/H[:,:,None]) * q1[:,:,None]**2/chi[:,:,None]**2 * P
    P_22_integ = (1/H[:,:,None]) * q2[:,:,None]**2/chi[:,:,None]**2 * P
    P_33_integ = (1/H[:,:,None]) * q3[:,:,None]**2/chi[:,:,None]**2 * P
    P_44_integ = (1/H[:,:,None]) * q4[:,:,None]**2/chi[:,:,None]**2 * P
    P_12_integ = (1/H[:,:,None]) * q1[:,:,None]*q2[:,:,None]/chi[:,:,None]**2 * P
    P_13_integ = (1/H[:,:,None]) * q1[:,:,None]*q3[:,:,None]/chi[:,:,None]**2 * P
    P_14_integ = (1/H[:,:,None]) * q1[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    P_23_integ = (1/H[:,:,None]) * q2[:,:,None]*q3[:,:,None]/chi[:,:,None]**2 * P
    P_24_integ = (1/H[:,:,None]) * q2[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    P_34_integ = (1/H[:,:,None]) * q3[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    
    P_proj_11 = jnp.trapezoid(P_11_integ, z_los, axis=1)
    P_proj_22 = jnp.trapezoid(P_22_integ, z_los, axis=1)
    P_proj_33 = jnp.trapezoid(P_33_integ, z_los, axis=1)
    P_proj_44 = jnp.trapezoid(P_44_integ, z_los, axis=1)
    P_proj_12 = jnp.trapezoid(P_12_integ, z_los, axis=1)
    P_proj_13 = jnp.trapezoid(P_13_integ, z_los, axis=1)
    P_proj_14 = jnp.trapezoid(P_14_integ, z_los, axis=1)
    P_proj_23 = jnp.trapezoid(P_23_integ, z_los, axis=1)
    P_proj_24 = jnp.trapezoid(P_24_integ, z_los, axis=1)
    P_proj_34 = jnp.trapezoid(P_34_integ, z_los, axis=1)

    pixwin_func = jnp.repeat(pixwin, theta.shape[0], axis=0)
    
    P_proj_11 *= pixwin_func**2 
    P_proj_22 *= pixwin_func**2
    P_proj_33 *= pixwin_func**2
    P_proj_44 *= pixwin_func**2
    P_proj_12 *= pixwin_func**2
    P_proj_13 *= pixwin_func**2
    P_proj_14 *= pixwin_func**2
    P_proj_23 *= pixwin_func**2
    P_proj_24 *= pixwin_func**2
    P_proj_34 *= pixwin_func**2
    
    return P_proj_11, P_proj_22, P_proj_33, P_proj_44, P_proj_12, P_proj_13, P_proj_14, P_proj_23, P_proj_24, P_proj_34

def los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp, iBamm, pixwin, z_list_integ, ell):
    z_los = jnp.repeat(jnp.reshape(z_list_integ, [1,-1, 1]), ell.size, axis=2)
    z_los = jnp.repeat(z_los, theta.shape[0], axis=0)
    
    iBapp_111_integ = (1/H[:,:,None]) * q1[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_112_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q2[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_113_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_114_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_122_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_123_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_124_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_133_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_134_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_144_integ = (1/H[:,:,None]) * q1[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_222_integ = (1/H[:,:,None]) * q2[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_223_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_224_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_233_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_234_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_244_integ = (1/H[:,:,None]) * q2[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_333_integ = (1/H[:,:,None]) * q3[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_334_integ = (1/H[:,:,None]) * q3[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_344_integ = (1/H[:,:,None]) * q3[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_444_integ = (1/H[:,:,None]) * q4[:,:,None]**3/chi[:,:,None]**4 * iBapp
    
    
    iBamm_111_integ = (1/H[:,:,None]) * q1[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_112_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q2[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_113_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_114_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_122_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_123_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_124_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_133_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_134_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_144_integ = (1/H[:,:,None]) * q1[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_222_integ = (1/H[:,:,None]) * q2[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_223_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_224_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_233_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_234_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_244_integ = (1/H[:,:,None]) * q2[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_333_integ = (1/H[:,:,None]) * q3[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_334_integ = (1/H[:,:,None]) * q3[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_344_integ = (1/H[:,:,None]) * q3[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_444_integ = (1/H[:,:,None]) * q4[:,:,None]**3/chi[:,:,None]**4 * iBamm
    
    iBapp_proj_111 = jnp.trapezoid(iBapp_111_integ, z_los, axis=1)
    iBapp_proj_112 = jnp.trapezoid(iBapp_112_integ, z_los, axis=1)
    iBapp_proj_113 = jnp.trapezoid(iBapp_113_integ, z_los, axis=1)
    iBapp_proj_114 = jnp.trapezoid(iBapp_114_integ, z_los, axis=1)
    iBapp_proj_122 = jnp.trapezoid(iBapp_122_integ, z_los, axis=1)
    iBapp_proj_123 = jnp.trapezoid(iBapp_123_integ, z_los, axis=1)
    iBapp_proj_124 = jnp.trapezoid(iBapp_124_integ, z_los, axis=1)
    iBapp_proj_133 = jnp.trapezoid(iBapp_133_integ, z_los, axis=1)
    iBapp_proj_134 = jnp.trapezoid(iBapp_134_integ, z_los, axis=1)
    iBapp_proj_144 = jnp.trapezoid(iBapp_144_integ, z_los, axis=1)
    iBapp_proj_222 = jnp.trapezoid(iBapp_222_integ, z_los, axis=1)
    iBapp_proj_223 = jnp.trapezoid(iBapp_223_integ, z_los, axis=1)
    iBapp_proj_224 = jnp.trapezoid(iBapp_224_integ, z_los, axis=1)
    iBapp_proj_233 = jnp.trapezoid(iBapp_233_integ, z_los, axis=1)
    iBapp_proj_234 = jnp.trapezoid(iBapp_234_integ, z_los, axis=1)
    iBapp_proj_244 = jnp.trapezoid(iBapp_244_integ, z_los, axis=1)
    iBapp_proj_333 = jnp.trapezoid(iBapp_333_integ, z_los, axis=1)
    iBapp_proj_334 = jnp.trapezoid(iBapp_334_integ, z_los, axis=1)
    iBapp_proj_344 = jnp.trapezoid(iBapp_344_integ, z_los, axis=1)
    iBapp_proj_444 = jnp.trapezoid(iBapp_444_integ, z_los, axis=1)
    
    iBamm_proj_111 = jnp.trapezoid(iBamm_111_integ, z_los, axis=1)
    iBamm_proj_112 = jnp.trapezoid(iBamm_112_integ, z_los, axis=1)
    iBamm_proj_113 = jnp.trapezoid(iBamm_113_integ, z_los, axis=1)
    iBamm_proj_114 = jnp.trapezoid(iBamm_114_integ, z_los, axis=1)
    iBamm_proj_122 = jnp.trapezoid(iBamm_122_integ, z_los, axis=1)
    iBamm_proj_123 = jnp.trapezoid(iBamm_123_integ, z_los, axis=1)
    iBamm_proj_124 = jnp.trapezoid(iBamm_124_integ, z_los, axis=1)
    iBamm_proj_133 = jnp.trapezoid(iBamm_133_integ, z_los, axis=1)
    iBamm_proj_134 = jnp.trapezoid(iBamm_134_integ, z_los, axis=1)
    iBamm_proj_144 = jnp.trapezoid(iBamm_144_integ, z_los, axis=1)
    iBamm_proj_222 = jnp.trapezoid(iBamm_222_integ, z_los, axis=1)
    iBamm_proj_223 = jnp.trapezoid(iBamm_223_integ, z_los, axis=1)
    iBamm_proj_224 = jnp.trapezoid(iBamm_224_integ, z_los, axis=1)
    iBamm_proj_233 = jnp.trapezoid(iBamm_233_integ, z_los, axis=1)
    iBamm_proj_234 = jnp.trapezoid(iBamm_234_integ, z_los, axis=1)
    iBamm_proj_244 = jnp.trapezoid(iBamm_244_integ, z_los, axis=1)
    iBamm_proj_333 = jnp.trapezoid(iBamm_333_integ, z_los, axis=1)
    iBamm_proj_334 = jnp.trapezoid(iBamm_334_integ, z_los, axis=1)
    iBamm_proj_344 = jnp.trapezoid(iBamm_344_integ, z_los, axis=1)
    iBamm_proj_444 = jnp.trapezoid(iBamm_444_integ, z_los, axis=1)

    pixwin_func = jnp.repeat(pixwin, theta.shape[0], axis=0)

    iBapp_proj_111 *= pixwin_func**2
    iBapp_proj_112 *= pixwin_func**2
    iBapp_proj_113 *= pixwin_func**2
    iBapp_proj_114 *= pixwin_func**2
    iBapp_proj_122 *= pixwin_func**2
    iBapp_proj_123 *= pixwin_func**2
    iBapp_proj_124 *= pixwin_func**2
    iBapp_proj_133 *= pixwin_func**2
    iBapp_proj_134 *= pixwin_func**2
    iBapp_proj_144 *= pixwin_func**2
    iBapp_proj_222 *= pixwin_func**2
    iBapp_proj_223 *= pixwin_func**2
    iBapp_proj_224 *= pixwin_func**2
    iBapp_proj_233 *= pixwin_func**2
    iBapp_proj_234 *= pixwin_func**2
    iBapp_proj_244 *= pixwin_func**2
    iBapp_proj_333 *= pixwin_func**2
    iBapp_proj_334 *= pixwin_func**2
    iBapp_proj_344 *= pixwin_func**2
    iBapp_proj_444 *= pixwin_func**2

    iBamm_proj_111 *= pixwin_func**2
    iBamm_proj_112 *= pixwin_func**2
    iBamm_proj_113 *= pixwin_func**2
    iBamm_proj_114 *= pixwin_func**2
    iBamm_proj_122 *= pixwin_func**2
    iBamm_proj_123 *= pixwin_func**2
    iBamm_proj_124 *= pixwin_func**2
    iBamm_proj_133 *= pixwin_func**2
    iBamm_proj_134 *= pixwin_func**2
    iBamm_proj_144 *= pixwin_func**2
    iBamm_proj_222 *= pixwin_func**2
    iBamm_proj_223 *= pixwin_func**2
    iBamm_proj_224 *= pixwin_func**2
    iBamm_proj_233 *= pixwin_func**2
    iBamm_proj_234 *= pixwin_func**2
    iBamm_proj_244 *= pixwin_func**2
    iBamm_proj_333 *= pixwin_func**2
    iBamm_proj_334 *= pixwin_func**2
    iBamm_proj_344 *= pixwin_func**2
    iBamm_proj_444 *= pixwin_func**2
    
    return iBapp_proj_111, iBapp_proj_112, iBapp_proj_113, iBapp_proj_114, iBapp_proj_122, iBapp_proj_123, iBapp_proj_124, iBapp_proj_133, iBapp_proj_134, iBapp_proj_144, iBapp_proj_222, iBapp_proj_223, iBapp_proj_224, iBapp_proj_233, iBapp_proj_234, iBapp_proj_244, iBapp_proj_333, iBapp_proj_334, iBapp_proj_344, iBapp_proj_444, iBamm_proj_111, iBamm_proj_112, iBamm_proj_113, iBamm_proj_114, iBamm_proj_122, iBamm_proj_123, iBamm_proj_124, iBamm_proj_133, iBamm_proj_134, iBamm_proj_144, iBamm_proj_222, iBamm_proj_223, iBamm_proj_224, iBamm_proj_233, iBamm_proj_234, iBamm_proj_244, iBamm_proj_333, iBamm_proj_334, iBamm_proj_344, iBamm_proj_444

#### Function computing H, chi and q from GP model ####
def get_H_chi_q_from_NN(theta):
    
    theta_H_chi_q = jnp.concat([jnp.reshape(theta[:,0],[-1,1]), jnp.reshape(theta[:,1],[-1,1])], axis=1)
    
    z_los_Wk = jnp.repeat(jnp.reshape(z_list_integ, [1,1,-1]), theta.shape[0], axis=0)
    z_los_Wk = jnp.repeat(z_los_Wk, n_z_los, axis=1)
    z_expansion = jnp.reshape(jnp.tile(z_list_integ, [theta.shape[0]]), [-1,1])
    theta_expansion = jnp.repeat(theta_H_chi_q, jnp.size(z_list_integ), axis=0)
    theta_expansion = jnp.concat([theta_expansion, z_expansion], axis=1)
    theta_expansion = jnp.astype(theta_expansion, jnp.float32)
    
    H, chi, D = get_H_chi_D(theta_expansion).T
    
    H = jnp.reshape(H, [theta.shape[0], n_z_los])
    H_0 = jnp.astype(jnp.repeat(jnp.reshape(theta[:,2], [-1,1]), n_z_los, axis=1), jnp.float32)
    H_0_cosmogrid = 67.3 #the fiducial current expansion rate of Cosmogrid
    H = H*H_0/H_0_cosmogrid
    
    chi = jnp.reshape(chi, [theta.shape[0], n_z_los])
    chi = chi*H_0_cosmogrid/H_0
    
    D = jnp.reshape(D, [theta.shape[0], n_z_los])

    H_0 = theta[:,2]/299792.458 
    Omega_m0 = theta[:,0]
    A_IA_0_NLA = theta[:,6]
    alpha_IA_0_NLA = jnp.zeros(theta.shape[0])
    
    H_0 = jnp.repeat(jnp.reshape(H_0, [-1,1]), n_z_los, axis=1)
    Omega_m0 = jnp.repeat(jnp.reshape(Omega_m0, [-1,1]), n_z_los, axis=1)
    A_IA_0_NLA = jnp.repeat(jnp.reshape(A_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    alpha_IA_0_NLA = jnp.repeat(jnp.reshape(alpha_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    
    # make two copies of chi as the chi_z and chi_zs components in the q kernel computation
    chi_z = jnp.copy(chi)
    chi_zs = jnp.copy(chi)
    
    chi_z = jnp.reshape(chi_z, [theta.shape[0], n_z_los, -1])
    chi_zs = jnp.reshape(chi_zs, [theta.shape[0], -1, n_z_los])
    chi_z = jnp.repeat(chi_z, n_z_los, axis=2)
    chi_zs = jnp.repeat(chi_zs, n_z_los, axis=1)
    
    theta_ns = theta
    n_zs_los_BIN1, n_zs_los_BIN2, n_zs_los_BIN3, n_zs_los_BIN4 = n_s_z_compute(theta_ns, z_list_integ, n_s_z_BIN1, n_s_z_BIN2, n_s_z_BIN3, n_s_z_BIN4)
    n_zs_los_BIN1_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN1, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN2_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN2, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN3_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN3, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN4_prime = jnp.repeat(jnp.reshape(n_zs_los_BIN4, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    
    indicator = jnp.ones((n_z_los, n_z_los))
    indicator = jnp.triu(indicator, 0)
    indicator = jnp.repeat(jnp.reshape(indicator, [-1, n_z_los, n_z_los]), theta.shape[0], axis=0)
    
    W1_integrand = n_zs_los_BIN1_prime * (chi_zs - chi_z)/chi_zs * indicator
    W2_integrand = n_zs_los_BIN2_prime * (chi_zs - chi_z)/chi_zs * indicator
    W3_integrand = n_zs_los_BIN3_prime * (chi_zs - chi_z)/chi_zs * indicator
    W4_integrand = n_zs_los_BIN4_prime * (chi_zs - chi_z)/chi_zs * indicator
    W1 = jnp.trapezoid(W1_integrand, z_los_Wk, axis=2)
    W2 = jnp.trapezoid(W2_integrand, z_los_Wk, axis=2)
    W3 = jnp.trapezoid(W3_integrand, z_los_Wk, axis=2)
    W4 = jnp.trapezoid(W4_integrand, z_los_Wk, axis=2)
    
    f_IA_NLA_z = -A_IA_0_NLA * 0.0134 * Omega_m0 * ((1.0 + z_list_integ[None,:])/1.62)**alpha_IA_0_NLA/D
    
    q1 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W1 + f_IA_NLA_z * H * n_zs_los_BIN1
    q2 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W2 + f_IA_NLA_z * H * n_zs_los_BIN2
    q3 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W3 + f_IA_NLA_z * H * n_zs_los_BIN3
    q4 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W4 + f_IA_NLA_z * H * n_zs_los_BIN4
    
    return H, chi, q1, q2, q3, q4


def correlation_compute(theta):
    
    theta_P = theta[:,:6].copy()
    theta_P = theta_P.at[:,2].multiply(0.01)
    z_expansion = jnp.reshape(jnp.tile(z_list_integ, [theta.shape[0]]), [-1,1])
    theta_expansion = jnp.repeat(theta_P, z_list_integ.size, axis=0)
    theta_expansion = jnp.concat([theta_expansion, z_expansion], axis=1)
    
    P = get_P_from_NN(theta_expansion)

    H, chi, q1, q2, q3, q4 = get_H_chi_q_from_NN(theta)
    
    P = jnp.reshape(P, [theta.shape[0], n_z_los, n_ell])
    
    P_proj_11, P_proj_22, P_proj_33, P_proj_44, P_proj_12, P_proj_13, P_proj_14, P_proj_23, P_proj_24, P_proj_34 = los_proj_power_spectrum(theta, q1, q2, q3, q4, chi, H, P)

    
    m_1 = theta[:,12]
    m_2 = theta[:,13]
    m_3 = theta[:,14]
    m_4 = theta[:,15]
    
    xip_11 = kitching_correction_and_FT_plus_xi(theta, P_proj_11)*(1.0+m_1[:,None])*(1.0+m_1[:,None])
    xip_22 = kitching_correction_and_FT_plus_xi(theta, P_proj_22)*(1.0+m_2[:,None])*(1.0+m_2[:,None])
    xip_33 = kitching_correction_and_FT_plus_xi(theta, P_proj_33)*(1.0+m_3[:,None])*(1.0+m_3[:,None])
    xip_44 = kitching_correction_and_FT_plus_xi(theta, P_proj_44)*(1.0+m_4[:,None])*(1.0+m_4[:,None])
    xip_12 = kitching_correction_and_FT_plus_xi(theta, P_proj_12)*(1.0+m_1[:,None])*(1.0+m_2[:,None])
    xip_13 = kitching_correction_and_FT_plus_xi(theta, P_proj_13)*(1.0+m_1[:,None])*(1.0+m_3[:,None])
    xip_14 = kitching_correction_and_FT_plus_xi(theta, P_proj_14)*(1.0+m_1[:,None])*(1.0+m_4[:,None])
    xip_23 = kitching_correction_and_FT_plus_xi(theta, P_proj_23)*(1.0+m_2[:,None])*(1.0+m_3[:,None])
    xip_24 = kitching_correction_and_FT_plus_xi(theta, P_proj_24)*(1.0+m_2[:,None])*(1.0+m_4[:,None])
    xip_34 = kitching_correction_and_FT_plus_xi(theta, P_proj_34)*(1.0+m_3[:,None])*(1.0+m_4[:,None])
    
    xim_11 = kitching_correction_and_FT_minus_xi(theta, P_proj_11)*(1.0+m_1[:,None])*(1.0+m_1[:,None])
    xim_22 = kitching_correction_and_FT_minus_xi(theta, P_proj_22)*(1.0+m_2[:,None])*(1.0+m_2[:,None])
    xim_33 = kitching_correction_and_FT_minus_xi(theta, P_proj_33)*(1.0+m_3[:,None])*(1.0+m_3[:,None])
    xim_44 = kitching_correction_and_FT_minus_xi(theta, P_proj_44)*(1.0+m_4[:,None])*(1.0+m_4[:,None])
    xim_12 = kitching_correction_and_FT_minus_xi(theta, P_proj_12)*(1.0+m_1[:,None])*(1.0+m_2[:,None])
    xim_13 = kitching_correction_and_FT_minus_xi(theta, P_proj_13)*(1.0+m_1[:,None])*(1.0+m_3[:,None])
    xim_14 = kitching_correction_and_FT_minus_xi(theta, P_proj_14)*(1.0+m_1[:,None])*(1.0+m_4[:,None])
    xim_23 = kitching_correction_and_FT_minus_xi(theta, P_proj_23)*(1.0+m_2[:,None])*(1.0+m_3[:,None])
    xim_24 = kitching_correction_and_FT_minus_xi(theta, P_proj_24)*(1.0+m_2[:,None])*(1.0+m_4[:,None])
    xim_34 = kitching_correction_and_FT_minus_xi(theta, P_proj_34)*(1.0+m_3[:,None])*(1.0+m_4[:,None])

    data_vector = jnp.concat([xip_11, xip_12, xip_13, xip_14, xip_22, xip_23, xip_24, xip_33, xip_34, xip_44, xim_11, xim_12, xim_13, xim_14, xim_22, xim_23, xim_24, xim_33, xim_34, xim_44], axis=1)
    
    return data_vector
