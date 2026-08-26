
"""
Conversion functions and property transformations

This module provides essential utility functions for converting between different
rock physics quantities, including seismic velocities, elastic moduli, petrophysical
parameters, and geophysical measurements. These conversions are fundamental to rock
physics workflow and modeling.
"""

import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .materials import FluidSet


def api_to_rho(api):
    """
    Convert oil API gravity to density.

    Parameters
    ----------
    api : float or ndarray
        API gravity (unitless).

    Returns
    -------
    rho : float or ndarray
        Density in grams per cubic centimeter (g/cc)
    """
    return 141.5 / (api + 131.5)

def rho_to_api(rho):
    """
    Convert density to oil API gravity.

    Parameters
    ----------
    rho : float or ndarray
        Density in grams per cubic centimeter (g/cc).

    Returns
    -------
    api : float or ndarray
        API gravity (unitless)
    """
    return 141.5/rho - 131.5

def psi_to_mpa(psi):
    """
    Convert pressure in pound per square inch (psi) into mega Pascal (MPa)
    
    Parameters
    ----------
    psi : float or ndarray
        Pressure value in psi

    Returns
    -------
    mpa : float or ndarray
        Pressure value in MPa
    """
    return 6894.76 * 1e-6 * psi

def mpa_to_psi(mpa):
    """Convert pressure in mega Pascal (MPa) into 
    
        Parameters
    ----------
    mpa : float or ndarray
        Pressure value in MPa

    Returns
    -------
    psi : float or ndarray
        Pressure value in psi
    """

    return 1e6/6894.76 * mpa

def velocity_to_shear ( s_velocity : np.ndarray, 
                        density    : np.ndarray) -> np.ndarray:
    """
    Compute shear modulus (G) from shear wave velocity and density.

    Parameters
    ----------
    s_velocity : ndarray
        Shear (S) wave velocity in meters per second (m/s).
    density : ndarray
        Bulk density in grams per cubic centimeter (g/cc).

    Returns
    -------
    shear : ndarray
        Shear modulus in gigapascals (GPa).

    """
    return s_velocity**2 * density / 1e6

def velocity_to_bulk ( p_velocity : np.ndarray, 
                       s_velocity : np.ndarray, 
                       density    : np.ndarray) -> np.ndarray:
    """
    Compute bulk modulus (K) from P- and S-wave velocities and density.

    Parameters
    ----------
    p_velocity : ndarray
        Compressional (P) wave velocity in meters per second (m/s).
    s_velocity : ndarray
        Shear (S) wave velocity in meters per second (m/s).
    density : ndarray
        Bulk density in grams per cubic centimeter (g/cc).

    Returns
    -------
    bulk : ndarray
        Bulk modulus in gigapascals (GPa).

    """
    return ((p_velocity**2 * density) -        \
            (4/3*s_velocity**2 * density))/1e6

def velocity_to_modulus ( p_velocity : np.ndarray, 
                          s_velocity : np.ndarray, 
                          density    : np.ndarray) -> tuple:
    """
    Compute bulk and shear moduli from seismic velocities and density.

    Parameters
    ----------
    p_velocity : ndarray
        P-wave velocity in m/s.
    s_velocity : ndarray
        S-wave velocity in m/s.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    (bulk, shear) : tuple of ndarray
        Bulk and shear moduli in GPa (K, G).
    """
    shear = velocity_to_shear (s_velocity, density)
    bulk = velocity_to_bulk (p_velocity, s_velocity, density)
    return bulk, shear

def velocity_to_poisson ( p_velocity : np.ndarray, 
                          s_velocity : np.ndarray) -> np.ndarray:
    """
    Compute Poisson's ratio from P- and S-wave velocities.

    Parameters
    ----------
    p_velocity : ndarray
        P-wave velocity in m/s.
    s_velocity : ndarray
        S-wave velocity in m/s.

    Returns
    -------
    poisson : ndarray
        Poisson's ratio (unitless)
    """
    return (p_velocity**2 - 2*s_velocity**2) / 2/(p_velocity**2 -s_velocity**2)

def modulus_to_poisson ( bulk : np.ndarray, 
                         shear : np.ndarray) -> np.ndarray:
    """
    Compute Poisson's ratio from bulk and shear moduli.

    Parameters
    ----------
    bulk : ndarray
        Bulk modulus in GPa.
    shear : ndarray
        Shear modulus in GPa.

    Returns
    -------
    poisson : ndarray
        Poisson's ratio (unitless)
    """
    return (3*bulk-2*shear)/(6*bulk+2*shear)

def velocity_to_lame ( p_velocity : np.ndarray, 
                       s_velocity : np.ndarray, 
                       density    : np.ndarray) -> np.ndarray:
    """
    Compute Lamé's first parameter (lambda) from velocities and density.

    Parameters
    ----------
    p_velocity : ndarray
        P-wave velocity in m/s.
    s_velocity : ndarray
        S-wave velocity in m/s.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    lame : ndarray
        Lamé's first parameter (lambda) in GPa
    """
    return density * (p_velocity**2 - 2*s_velocity**2) / 1e6

def velocity_to_young ( p_velocity : np.ndarray, 
                        s_velocity : np.ndarray, 
                        density    : np.ndarray) -> np.ndarray:
    """
    Compute Young's modulus (E) from seismic velocities and density.

    Parameters
    ----------
    p_velocity : ndarray
        P-wave velocity in m/s.
    s_velocity : ndarray
        S-wave velocity in m/s.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    young : ndarray
        Young's modulus in GPa
    """
    return density * s_velocity**2                  \
        * (3*p_velocity**2 - 4*s_velocity**2)       \
        / (p_velocity**2 - s_velocity**2) / 1e6

def modulus_to_young ( bulk : np.ndarray, 
                       shear : np.ndarray) -> np.ndarray:
    """
    Compute Young's modulus from bulk and shear moduli.

    Parameters
    ----------
    bulk : ndarray
        Bulk modulus in GPa.
    shear : ndarray
        Shear modulus in GPa.

    Returns
    -------
    young : ndarray
        Young's modulus in GPa
    """
    return 9*bulk*shear/(3*bulk + shear)

def velocity_ratio ( p_velocity : np.ndarray, 
                     s_velocity : np.ndarray) -> np.ndarray:
    """
    Compute the ratio of P- to S-wave velocity.

    Parameters
    ----------
    p_velocity : ndarray
        P-wave velocity in m/s.
    s_velocity : ndarray
        S-wave velocity in m/s.

    Returns
    -------
    ratio : ndarray
        Velocity ratio Vp/Vs (unitless).
    """
    return (p_velocity/s_velocity)

def velratio_to_poisson ( velocity_ratio : np.ndarray) -> np.ndarray:
    """
    Convert P- to S-wave Velocity ratio (Vp/Vs) to Poisson's ratio.

    Parameters
    ----------
    velocity_ratio : ndarray
        P- to S-wave Velocity ratio, Vp/Vs (unitless).

    Returns
    -------
    poisson : ndarray
        Poisson's ratio (unitless), computed from ratio r:
        nu = (r^2 - 2) / (2 r^2 - 2)
    """
    return ((velocity_ratio**2 - 2) / (2*velocity_ratio**2 - 2))

def poisson_to_velratio ( poisson : np.ndarray) -> np.ndarray:
    """
    Convert Poisson's ratio to P- to S-wave Velocity ratio (Vp/Vs).

    Parameters
    ----------
    poisson : ndarray
        Poisson's ratio (unitless).

    Returns
    -------
    velocity_ratio : ndarray
        P- to S-wave Velocity ratio, Vp/Vs (unitless)
    """
    return np.sqrt((2*poisson - 2) / (2*poisson - 1))

def modulus_to_svelocity ( shear : np.ndarray, 
                           density : np.ndarray) -> np.ndarray:
    """
    Compute S-wave velocity from shear modulus and density.

    Parameters
    ----------
    shear : ndarray
        Shear modulus in GPa.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    s_velocity : ndarray
        S-wave velocity in meters per second (m/s). Uses
        Vs = sqrt(G / rho) with appropriate unit conversions.
    """
    return np.sqrt(shear / density) * 1e3

def modulus_to_pvelocity ( bulk  : np.ndarray, 
                           shear : np.ndarray, 
                           density : np.ndarray) -> np.ndarray:
    """
    Compute P-wave velocity from bulk and shear moduli and density.

    Parameters
    ----------
    bulk : ndarray
        Bulk modulus in GPa.
    shear : ndarray
        Shear modulus in GPa.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    p_velocity : ndarray
        P-wave velocity in meters per second (m/s)
    """
    return np.sqrt((bulk + 4/3*shear) / density) * 1e3

def modulus_to_velocity ( bulk  : np.ndarray, 
                          shear : np.ndarray, 
                          density : np.ndarray) -> tuple:
    """
    Compute P- and S-wave velocities from bulk and shear moduli and density.

    Parameters
    ----------
    bulk : ndarray
        Bulk modulus in GPa.
    shear : ndarray
        Shear modulus in GPa.
    density : ndarray
        Density in g/cc.

    Returns
    -------
    (p_velocity, s_velocity) : tuple of ndarray
        P- and S-wave velocities in m/s.
    """
    s_velocisty = modulus_to_svelocity(shear, density)
    p_velocisty = modulus_to_pvelocity(bulk, shear, density)
    return p_velocisty, s_velocisty

def modulus_to_Lame(bulk, 
                    shear):
    """
    Compute Lamé's first parameter lambda from bulk and shear moduli.

    Parameters
    ----------
    bulk : ndarray or float
        Bulk modulus in GPa.
    shear : ndarray or float
        Shear modulus in GPa.

    Returns
    -------
    lame : ndarray or float
        Lamé's first parameter (lambda) in GPa, lambda = K - 2/3 G.
    """
    return bulk-2/3*shear

def modulus_to_stiffness(bulk, shear):
    """
    Assemble isotropic stiffness tensor (Voigt notation) components from moduli.

    Parameters
    ----------
    bulk : ndarray or float
        Bulk modulus in GPa.
    shear : ndarray or float
        Shear modulus in GPa.

    Returns
    -------
    c11, c33, c44, c66, c12, c13 : tuple
        Independent stiffness coefficients in GPa for an isotropic material:
        c11 = c33 = lambda + 2 G, c44 = c66 = G, c12 = c13 = lambda.
    """
    lame = modulus_to_Lame(bulk, shear)

    c11 = lame + 2*shear
    c33 = c11
    c44 = shear
    c66 = c44
    c12 = lame
    c13 = c12
    return c11, c33, c44, c66, c12, c13

def total_to_effective_fluidset(fluidset: "FluidSet", 
                                total_porosity: float | np.ndarray, 
                                effective_porosity: float | np.ndarray):
    """
    Convert a fluid set defined on total porosity to an effective-porosity basis.

    Parameters
    ----------
    fluidset : FluidSet
        FluidSet object
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    effective_fluidset : FluidSet
        FluidSet object
        the effective-porosity basis. Non-bound components is treated as effective.
    """
    effective_fluidset = deepcopy(fluidset)
    for fluid in effective_fluidset.component_set.keys():
        if fluid == fluidset.bound_fluid:
            effective_fluidset.component_set[fluid] = total_to_effective_saturation_bound(fluidset.component_set[fluid], total_porosity, effective_porosity)
        else:
            effective_fluidset.component_set[fluid] = total_to_effective_saturation_nonbound(fluidset.component_set[fluid], total_porosity, effective_porosity)
    return effective_fluidset

def total_to_effective_saturation_bound(total_saturation: float | np.ndarray, 
                                        total_porosity: float | np.ndarray, 
                                        effective_porosity: float | np.ndarray):
    """
    Convert total saturation of a bound (immobile) fluid to effective saturation.

    Parameters
    ----------
    total_saturation : float or ndarray
        Saturation of the fluid in total pore volume (fraction, 0-1).
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    effective_saturation : float or ndarray
        Saturation fraction within the effective porosity (0-1).
    """
    bound_fluid_porosity = total_porosity - effective_porosity
    return (total_saturation*total_porosity - bound_fluid_porosity) / effective_porosity

def total_to_effective_saturation_nonbound( total_saturation: float | np.ndarray, 
                                            total_porosity: float | np.ndarray, 
                                            effective_porosity: float | np.ndarray):
    """
    Convert total saturation of a non-bound fluid to effective saturation.

    Parameters
    ----------
    total_saturation : float or ndarray
        Saturation of the fluid in total pore volume (fraction, 0-1).
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    effective_saturation : float or ndarray
        Saturation fraction within the effective porosity (0-1).
    """
    return total_saturation*total_porosity/effective_porosity

def shale_to_clay_fraction(shale_fraction: float | np.ndarray, 
                           total_porosity: float | np.ndarray, 
                           effective_porosity: float | np.ndarray):
    """
    Convert a bulk shale fraction to an equivalent clay volume fraction.

    Parameters
    ----------
    shale_fraction : float or ndarray
        Fractional shale content by volume in bulk rock (0-1).
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    clay_volume : float or ndarray
        Clay (solid-phase) volume fraction relative to bulk rock (0-1).

    Notes
    -----
    This converts a bulk shale fraction to clay volume assuming that differences
    between total and effective porosity correspond to the shale bound fluid volume.
    """
    clay_volume = ((1-effective_porosity)*shale_fraction - (total_porosity-effective_porosity)) / (1-total_porosity)
    return clay_volume

def clay_to_shale_fraction(clay_fraction: float | np.ndarray, 
                           total_porosity: float | np.ndarray, 
                           effective_porosity: float | np.ndarray):
    """
    Convert a clay solid-phase fraction to corresponding shale fraction.

    Parameters
    ----------
    clay_fraction : float or ndarray
        Clay fraction in the solid (0-1).
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    shale_fraction : float or ndarray
        Bulk shale fraction by volume (0-1).

    Notes
    -----
    This converts a shale fraction to corresponding clay volume assuming that differences
    between total and effective porosity correspond to the shale bound fluid volume.
    """
    clay_fraction = ((1-total_porosity)*clay_fraction + (total_porosity-effective_porosity)) / (1-effective_porosity)
    return clay_fraction

def effective_to_total_fraction( non_clay_fraction: float | np.ndarray, 
                                 total_porosity: float | np.ndarray, 
                                 effective_porosity: float | np.ndarray):
    """
    Convert a non-clay fraction defined on effective-porosity solids to the total-rock basis.

    Parameters
    ----------
    non_clay_fraction : float or ndarray
        Fraction of non-clay material (solid fraction) defined per unit of effective solid volume (0-1).
    total_porosity : float or ndarray
        Total porosity (fraction, 0-1).
    effective_porosity : float or ndarray
        Effective porosity (fraction, 0-1).

    Returns
    -------
    total_fraction : float or ndarray
        Fraction of non-clay material on the total-rock solid volume basis (0-1).
    """
    return non_clay_fraction*(1-effective_porosity)/(1-total_porosity)

def vint_to_vrms(Vint, t):
    """
    Convert interval velocity to RMS velocity.

    Parameters
    ----------
    Vint : array_like
        Interval velocity vector, shape (N,).
    t : array_like
        Time vector, shape (N,).

    Returns
    -------
    Vrms : ndarray
        RMS velocity vector, shape (N,).
    """
    dt = np.empty_like(t)
    dt[0] = t[0]
    dt[1:] = np.diff(t)

    Vrms = np.sqrt(np.cumsum(Vint**2 * dt)) / t

    return Vrms

def vrms_to_vint(Vrms, t):
    """
    Convert RMS velocity to interval velocity.

    Parameters
    ----------
    Vrms : array_like
        RMS velocity vector, shape (N,).
    t : array_like
        Time vector, shape (N,).

    Returns
    -------
    Vint : ndarray
        Interval velocity vector, shape (N,).
    """
    dt = np.empty_like(t)
    dt[0] = t[0]
    dt[1:] = np.diff(t)

    Vint = np.sqrt(np.cumsum(Vrms**2 * dt)) / t

    return Vint

def vint_to_vavg(Vint, t):
    """
    Convert interval velocity to average velocity.

    Parameters
    ----------
    Vint : array_like
        Interval velocity vector, shape (N,).
    t : array_like
        Time vector, shape (N,).

    Returns
    -------
    Vavg : ndarray
        Average velocity vector, shape (N,).
    """

    dt = np.empty_like(t)
    dt[0] = t[0]
    dt[1:] = np.diff(t)

    Vavg = np.cumsum(Vint * dt) / t

    return Vavg

def vavg_to_vint(Vavg, t):
    """
    Convert average velocity to interval velocity.

    Parameters
    ----------
    Vavg : array_like
        Average velocity vector, shape (N,).
    t : array_like
        Time vector, shape (N,).

    Returns
    -------
    Vint : ndarray
        Interval velocity vector, shape (N,).
    """

    dt = np.empty_like(t)
    dt[0] = t[0]
    dt[1:] = np.diff(t)

    Vint = np.cumsum(Vavg * dt) / t

    return Vint

def depth_to_time(depth, Vint):
    """
    Convert depth to time using interval velocity model.

    Parameters
    ----------
    depth : array_like
        Depth vector, shape (N,).
    Vint : array_like
        Interval velocity vector, shape (N,).

    Returns
    -------
    time : ndarray
        Time vector, shape (N,).
    """

    dz = np.empty_like(depth)
    dz[0] = depth[0]
    dz[1:] = np.diff(depth)

    time = 2 * np.cumsum(dz / Vint)

    return time 

def time_to_depth(time, Vint):
    """
    Convert time to depth using interval velocity model.

    Parameters
    ----------
    time : array_like
        Time vector, shape (N,).
    Vint : array_like
        Interval velocity vector, shape (N,).

    Returns
    -------
    depth : ndarray
        Depth vector, shape (N,).
    """

    time = np.asarray(time)
    Vint = np.asarray(Vint)
    dt = np.empty_like(time)
    dt[0] = time[0]
    dt[1:] = np.diff(time)

    depth = 0.5 * np.cumsum(Vint * dt)

    return depth

def regular_resample(x, y, dx):
    """
    Resample data (x, y) onto a regular grid with spacing dx using linear interpolation.

    Parameters
    ----------
    x : np.ndarray
        Original x-coordinates, shape (N,).
    y : np.ndarray
        Original y-values, shape (N,).
    dx : float
        Desired spacing for the regular grid.

    Returns
    -------
    x_reg : ndarray
        Regularly spaced x-coordinates.
    y_reg : ndarray
        Interpolated y-values on the regular grid.
    """

    x_reg = np.arange(np.min(x), np.max(x), dx)
    y_reg = np.interp(x_reg, x, y)

    return y_reg, x_reg

def stiffness_to_thomsen(c11, c33, c44, c66, c13, rho=None):
    """
    Converts the Voigt stiffness elements of TI media into Thomsen's parameters.

    Parameters
    ----------
    c11, c33, c44, c66, c13 : array_like
        Elements of the Voigt stiffness matrix (in GPa).
    rho : array_like, optional
        Density in g/cc. If provided, Vp0 and Vs0 are computed.

    Returns
    -------
    epsilon : ndarray
        Thomsen's epsilon parameter.
    gamma : ndarray
        Thomsen's gamma parameter.
    delta : ndarray
        Thomsen's delta parameter.
    Vp0 : ndarray
        P-wave velocity along the symmetry axis (m/s). Zero if `rho` not provided.
    Vs0 : ndarray
        S-wave velocity along the symmetry axis (m/s). Zero if `rho` not provided.

    """
    epsilon = (c11 - c33) / (2 * c33)
    gamma = (c66 - c44) / (2 * c44)
    delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2 * c33 * (c33 - c44))

    if rho is not None:
        rho = np.asarray(rho)
        Vp0 = 1000 * np.sqrt(c33 / rho)
        Vs0 = 1000 * np.sqrt(c44 / rho)
    else:
        Vp0 = np.zeros_like(c33)
        Vs0 = np.zeros_like(c44)

    return epsilon, gamma, delta, Vp0, Vs0

def stiffness_to_tensor(c11, c33, c44, c66, c12, c13):
    """
    Converts the Voigt stiffness elements of TI media into elastic stiffness tensor.

    Parameters
    ----------
    c11, c33, c44, c66, c12, c13 : float or array_like
        Elements of the Voigt stiffness matrix for transversely isotropic media.

    Returns
    -------
    Cijkl : ndarray
        3x3x3x3 elastic stiffness tensor.

    """

    I,J,K,L = symmetry_to_tensorindex('21')
    P = tensorindex_to_voigtindex(I, J)
    Q = tensorindex_to_voigtindex(K, L)    
    Cijkl = np.zeros((9, 9))
    idx = (P==1) and (Q==1)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c11, '21')
    idx = (P==3) and (Q==3)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c33, '21')
    idx = (P==4) and (Q==4)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c44, '21')
    idx = (P==6) and (Q==6)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c66, '21')
    idx = (P==1) and (Q==3)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c13, '21')
    idx = (P==1) and (Q==2)
    Cijkl = _set_value_to_tensor(Cijkl, np.column_stack([I[idx], J[idx], K[idx], L[idx]]), c12, '21')

    return Cijkl

def stiffness_to_velocity(c11, c33, c44, c66, c13, rho, theta):
    """
    P- and S-velocities of TI media in a given direction based on Voigt stiffness elements.

    Parameters
    ----------
    c11, c33, c44, c66, c13 : array_like
        Voigt stiffness elements in GPa.
    rho : array_like
        Density in g/cc.
    theta : array_like
        Angle between wave propagation direction and TI symmetry axis, in degrees.

    Returns
    -------
    Vp : ndarray
        Compressional (P-wave) velocity in m/s.
    Vsv : ndarray
        Pseudo-shear (SV-wave) velocity in m/s.
    Vsh : ndarray
        Pure shear (SH-wave) velocity in m/s.

    References
    ----------
    Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
    Cambridge University Press.

    """
    import numpy as np

    theta_rad = np.deg2rad(theta)
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    sin_2t = np.sin(2 * theta_rad)

    M = ((c11 - c44) * sin_t**2 - (c33 - c44) * cos_t**2)**2 + (c13 + c44)**2 * sin_2t**2

    Vp = 1000 * np.sqrt(c11 * sin_t**2 + c33 * cos_t**2 + c44 + np.sqrt(M)) / np.sqrt(2 * rho)
    Vsv = 1000 * np.sqrt(c11 * sin_t**2 + c33 * cos_t**2 + c44 - np.sqrt(M)) / np.sqrt(2 * rho)
    Vsh = 1000 * np.sqrt(c66 * sin_t**2 + c44 * cos_t**2) / np.sqrt(rho)

    return Vp, Vsv, Vsh

def tensorindex_to_voigtindex(I, J):
    """
    Converts the stiffness tensor indices into corresponding Voigt stiffness matrix index.

    Parameters
    ----------
    I : array_like
        First index of the tensor (1-based).
    J : array_like
        Second index of the tensor (1-based).

    Returns
    -------
    P : ndarray
        Voigt index corresponding to (I, J), following below the mapping:
        (1,1)->1, (2,2)->2, (3,3)->3, (2,3)/(3,2)->4, (1,3)/(3,1)->5, (1,2)/(2,1)->6.
    """
    I = np.asarray(I)
    J = np.asarray(J)

    delta = (I == J).astype(int)
    P = I * delta + (1 - delta) * (9 - I - J)

    return P

def symmetry_to_tensorindex(symmetry: Literal['45', '36', '21', '0']):
    """
    Create four vectors of indices for the tensor of given symmetry.

    Parameters
    ----------
    symmetry : str
        Symmetry type based on number of independent components:
        - '45': Gijkl = Gjilk (simple matrix symmetry)
        - '36': Gijkl = Gjikl = Gijlk = Gjilk
        - '21': Gijkl = Gklij (in addition to '36' symmetries)
        - '0': No symmetry

    Returns
    -------
    I, J, K, L : ndarray
        Each of shape (81,) for '0', '36', or (9, 9) for '21',
        containing tensor indices (1-based) corresponding to the symmetry.

    Author
    ------
    Mostafa Abbasi, Ph.D., Geophysics
    email: abbasi.mstfa@gmail.com
    Last revision: 17-August-2022
    Copyright (c) 2022, Mostafa Abbasi
    """
    if symmetry == '0':
        i = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
        j = np.array([[1, 2, 3],
                    [1, 2, 3],
                    [1, 3, 3]])
    elif symmetry in ('36', '21'):
        i = np.array([[1, 1, 1],
                    [1, 2, 2],
                    [1, 2, 3]])
        j = np.array([[1, 2, 3],
                    [2, 2, 3],
                    [3, 3, 3]])
    else:
        raise ValueError("Unsupported symmetry type")

    u = np.ones(3, dtype=int)
    I = np.kron(i, u).flatten()
    J = np.kron(j, u).flatten()
    K = np.kron(u, i).flatten()
    L = np.kron(u, j).flatten()

    if symmetry == '21':
        PQRS = np.column_stack([I, J, K, L])
        # Enforce major symmetry: G_ijkl = G_klij
        for c1 in range(81):
            pq = PQRS[c1, :2]
            rs = PQRS[c1, 2:]
            for c2 in range(81):
                if np.array_equal(PQRS[c2, :], np.concatenate([rs, pq])):
                    PQRS[c2, :] = np.concatenate([pq, rs])
        I = PQRS[:, 0].reshape((9, 9))
        J = PQRS[:, 1].reshape((9, 9))
        K = PQRS[:, 2].reshape((9, 9))
        L = PQRS[:, 3].reshape((9, 9))
        # Flatten again to match MATLAB's output style in indexing context
        I = I.flatten()
        J = J.flatten()
        K = K.flatten()
        L = L.flatten()

    return I, J, K, L

def _set_value_to_tensor(G, pqrs, value, symmetry: Literal['45', '36', '21', '0']):
    """
    Set a value to the specified element of tensor G.

    Parameters
    ----------
    G : ndarray
        Input tensor as a 9x9 matrix.
    pqrs : array_like
        Array of shape (nvalues, 4) containing tensor indices (1-based).
    value : array_like
        Array of shape (nvalues,) containing values to assign.
    symmetry : str
        Symmetry type:
        - '45': Gijkl = Gjilk (simple matrix symmetry)
        - '36': Gijkl = Gjikl = Gijlk = Gjilk
        - '21': Gijkl = Gklij (in addition to '36' symmetries)
        - '0': No symmetry

    Returns
    -------
    G : ndarray
        Updated 9x9 tensor matrix.

    """

    if G.shape != (9, 9):
        raise ValueError('Tensor G must be a 9-by-9 matrix')

    pqrs = np.atleast_2d(pqrs)
    value = np.atleast_1d(value)

    if symmetry == '21':
        # Generate full index mapping for 21-component symmetry (TI)
        u = np.ones(3, dtype=int)
        i = np.array([[1, 1, 1],
                    [1, 2, 2],
                    [1, 2, 3]], dtype=int)
        j = np.array([[1, 2, 3],
                    [2, 2, 3],
                    [3, 3, 3]], dtype=int)

        I = np.kron(i, u).flatten()
        J = np.kron(j, u).flatten()
        K = np.kron(u, i).flatten()
        L = np.kron(u, j).flatten()

        PQRS = np.column_stack([I, J, K, L])

        # Enforce minor and major symmetries: G_ijkl = G_klij
        for c1 in range(81):
            pq = PQRS[c1, :2]
            rs = PQRS[c1, 2:]
            for c2 in range(81):
                if np.array_equal(PQRS[c2, :], np.concatenate([rs, pq])):
                    PQRS[c2, :] = np.concatenate([pq, rs])

        I_map = PQRS[:, 0].reshape((9, 9))
        J_map = PQRS[:, 1].reshape((9, 9))
        K_map = PQRS[:, 2].reshape((9, 9))
        L_map = PQRS[:, 3].reshape((9, 9))

        for m in range(len(value)):
            p, q, r, s = pqrs[m]
            mask = (I_map == p) and (J_map == q) and (K_map == r) and (L_map == s)
            G[mask] = value[m]

    elif symmetry in ('36', '21'):
        # For '36' and '21', same indexing logic applies using precomputed maps
        # Since '21' already handled, this clause only triggers for '36'
        # But we lack I,J,K,L for '36' alone, so assume standard Voigt minor symmetry
        # We'll treat '36' same as '21' without major symmetry
        raise NotImplementedError("Symmetry type '36' is not implemented in this conversion.")

    elif symmetry == '0':
        for m in range(len(value)):
            p, q, r, s = pqrs[m]
            G[3 * (p - 1) + r, 3 * (q - 1) + s] = value[m]

    elif symmetry == '45':
        for m in range(len(value)):
            p, q, r, s = pqrs[m]
            idx1 = (3 * (p - 1) + r, 3 * (q - 1) + s)
            idx2 = (3 * (q - 1) + s, 3 * (p - 1) + r)
            G[idx1] = value[m]
            G[idx2] = value[m]

    else:
        raise ValueError("Unsupported symmetry type")

    return G




