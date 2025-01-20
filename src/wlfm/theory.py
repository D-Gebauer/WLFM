import jax.numpy as jnp
from jax.scipy import interpolate


#### A scipy function which compute n(z) w.r.t different photometric redshift uncertainty ####

def n_s_z_compute(theta, z_list_integ, n_s_z_BIN1, n_s_z_BIN2, n_s_z_BIN3, n_s_z_BIN4):
    
    z = jnp.repeat(z_list_integ.reshape(1,-1), theta.shape[0], axis=0)
    delta_z_1 = theta[:,5]
    delta_z_2 = theta[:,6]
    delta_z_3 = theta[:,7]
    delta_z_4 = theta[:,8]
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

def kitching_correction_and_FT_plus_zeta(theta, spectrum, FT_plus_kernel, n_angular_bins, ell, fullell, fullell_kitching):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_plus_kernel[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_minus_zeta(theta, spectrum, FT_minus_kernel, n_angular_bins, ell, fullell, fullell_kitching):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_minus_kernel[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_plus_xi(theta, spectrum, FT_plus_binave_xi, n_angular_bins, ell, fullell, fullell_kitching):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_plus_binave_xi[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

def kitching_correction_and_FT_minus_xi(theta, spectrum, FT_minus_binave_xi, n_angular_bins, ell, fullell, fullell_kitching):
     
    interp_spectrum = interpolation(ell, spectrum, fullell_kitching)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell + 2.0)*(fullell + 1.0)*fullell*(fullell - 1.0)*interp_spectrum/(fullell + 0.5)**4
    spectrum_kitching = jnp.repeat(jnp.reshape(spectrum_kitching, [theta.shape[0], fullell.size, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell[None,:,None] + 1.0)/(2*jnp.pi)/(fullell[None,:,None]*(fullell[None,:,None]+1))**2 * FT_minus_binave_xi[None,:,:] * spectrum_kitching
    CF = jnp.sum(CF, axis=1)
    return CF

#### LOS projection functions ####
def los_proj_power_spectrum(theta, q1, q2, q3, q4, chi, H, P, pixwin, z_list_integ, ell):
    z_los_tf = jnp.repeat(jnp.reshape(z_list_integ, [1, -1, 1]), ell.size, axis=2)
    z_los_tf = jnp.repeat(z_los_tf, theta.shape[0], axis=0)
    
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
    
    P_proj_11 = jnp.trapezoid(P_11_integ, z_los_tf, axis=1)
    P_proj_22 = jnp.trapezoid(P_22_integ, z_los_tf, axis=1)
    P_proj_33 = jnp.trapezoid(P_33_integ, z_los_tf, axis=1)
    P_proj_44 = jnp.trapezoid(P_44_integ, z_los_tf, axis=1)
    P_proj_12 = jnp.trapezoid(P_12_integ, z_los_tf, axis=1)
    P_proj_13 = jnp.trapezoid(P_13_integ, z_los_tf, axis=1)
    P_proj_14 = jnp.trapezoid(P_14_integ, z_los_tf, axis=1)
    P_proj_23 = jnp.trapezoid(P_23_integ, z_los_tf, axis=1)
    P_proj_24 = jnp.trapezoid(P_24_integ, z_los_tf, axis=1)
    P_proj_34 = jnp.trapezoid(P_34_integ, z_los_tf, axis=1)

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
def get_H_chi_q_from_GP(theta, z_list_integ, n_z_los, emulator_H, emulator_chi, emulator_D, H_mean, H_std, chi_mean, chi_std, D_mean, D_std, n_s_z_BIN1, n_s_z_BIN2, n_s_z_BIN3, n_s_z_BIN4):
    
    theta_H_chi_q = jnp.concat([jnp.reshape(theta[:,0],[-1,1]), jnp.reshape(theta[:,2],[-1,1])], axis=1) # our emulators here only require Om and w0
    z_los_Wk_tf = jnp.repeat(jnp.reshape(z_list_integ, [1,1,-1]), theta.shape[0], axis=0)
    z_los_Wk_tf = jnp.repeat(z_los_Wk_tf, n_z_los, axis=1)
    z_expansion = jnp.reshape(jnp.tile(z_list_integ, [theta.shape[0]]), [-1,1])
    theta_expansion = jnp.repeat(theta_H_chi_q, jnp.size(z_list_integ), axis=0)
    theta_expansion = jnp.concat([theta_expansion, z_expansion], axis=1)
    theta_expansion = jnp.astype(theta_expansion, dtype=jnp.float64)
    
    H = jnp.astype(emulator_H.predict_f_compiled(theta_expansion)[0], dtype=jnp.float32)
    H = jnp.reshape(H * H_std + H_mean, [theta.shape[0], n_z_los])
    h_0 = jnp.astype(jnp.repeat(jnp.reshape(theta[:,3], [-1,1]), n_z_los, axis=1), dtype=jnp.float32)
    h_cosmogrid = 0.673 #the fiducial current expansion rate of Cosmogrid
    H = H*h_0/h_cosmogrid
    
    chi = jnp.astype(emulator_chi.predict_f_compiled(theta_expansion)[0], dtype=jnp.float32)
    chi = jnp.reshape(chi * chi_std + chi_mean, [theta.shape[0], n_z_los])
    chi = chi*h_cosmogrid/h_0
    
    D = jnp.astype(emulator_D.predict_f_compiled(theta_expansion)[0], dtype=jnp.float32)
    D = jnp.reshape(D * D_std + D_mean, [theta.shape[0], n_z_los])

    H_0 = theta[:,3]*100/299792.458 
    Omega_m0 = theta[:,0]
    A_IA_0_NLA = theta[:,13]
    #alpha_IA_0_NLA = theta[:,10]
    alpha_IA_0_NLA = jnp.zeros(theta.shape[0])
    
    H_0 = jnp.repeat(jnp.reshape(H_0, [-1,1]), n_z_los, axis=1)
    Omega_m0 = jnp.repeat(jnp.reshape(Omega_m0, [-1,1]), n_z_los, axis=1)
    A_IA_0_NLA = jnp.repeat(jnp.reshape(A_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    alpha_IA_0_NLA = jnp.repeat(jnp.reshape(alpha_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    
    # make two copies of chi as the chi_z and chi_zs components in the q kernel computation
    chi_z = jnp.identity(chi)
    chi_zs = jnp.identity(chi)
    
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
    indicator = jnp.linalg.band_part(indicator, 0, -1)
    indicator = jnp.repeat(jnp.reshape(indicator, [-1, n_z_los, n_z_los]), theta.shape[0], axis=0)
    
    W1_integrand = n_zs_los_BIN1_prime * (chi_zs - chi_z)/chi_zs * indicator
    W2_integrand = n_zs_los_BIN2_prime * (chi_zs - chi_z)/chi_zs * indicator
    W3_integrand = n_zs_los_BIN3_prime * (chi_zs - chi_z)/chi_zs * indicator
    W4_integrand = n_zs_los_BIN4_prime * (chi_zs - chi_z)/chi_zs * indicator
    W1 = jnp.trapezoid(W1_integrand, z_los_Wk_tf, axis=2)
    W2 = jnp.trapezoid(W2_integrand, z_los_Wk_tf, axis=2)
    W3 = jnp.trapezoid(W3_integrand, z_los_Wk_tf, axis=2)
    W4 = jnp.trapezoid(W4_integrand, z_los_Wk_tf, axis=2)
    
    f_IA_NLA_z = -A_IA_0_NLA * 0.0134 * Omega_m0 * ((1.0 + z_list_integ[None,:])/1.62)**alpha_IA_0_NLA/D
    
    q1 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W1 + f_IA_NLA_z * H * n_zs_los_BIN1
    q2 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W2 + f_IA_NLA_z * H * n_zs_los_BIN2
    q3 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W3 + f_IA_NLA_z * H * n_zs_los_BIN3
    q4 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ) * chi * W4 + f_IA_NLA_z * H * n_zs_los_BIN4
    
    return H, chi, q1, q2, q3, q4