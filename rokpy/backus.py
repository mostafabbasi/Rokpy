import numpy as np
from rokpy.conversions import stiffness_to_thomsen, stiffness_to_velocity, velocity_to_modulus, modulus_to_stiffness
from scipy.interpolate import interp1d

def backus_avgerage(Vp, Vs, Rho, z, dz, interpmethod='linear'):
    """
    Backus average of given sets of Vp, Vs and Rho logs.

    Parameters
    ----------
    Vp : array_like
        P-wave velocity, shape (nsamples,).
    Vs : array_like
        S-wave velocity, shape (nsamples,).
    Rho : array_like
        Density (g/cc), shape (nsamples,).
    z : array_like
        Sample depths, shape (nsamples,).
    dz : float
        Length of Backus averaging window (same unit as z).
    interpmethod : str, optional
        Interpolation method after upscaling. Default is 'linear'.
        If 'none', no interpolation is performed and output size differs from input.
        Supported methods: 'linear', 'nearest', 'next', 'previous', 'pchip',
        'cubic', 'v5cubic', 'makima', 'spline'.

    Returns
    -------
    Vp0up : ndarray
        Up-scaled P-velocity along symmetry axis.
    Vs0up : ndarray
        Up-scaled S-velocity along symmetry axis.
    Rhoup : ndarray
        Up-scaled density.
    epsilon : ndarray
        Thomsen's epsilon parameter.
    gamma : ndarray
        Thomsen's gamma parameter.
    delta : ndarray
        Thomsen's delta parameter.
    zup : ndarray
        Output depth vector (same as input `z` if interpolation is applied;
        otherwise, coarser grid).
    """
    if dz < np.max(np.diff(z)):
        raise ValueError('New scale cannot be finer than original scale')

    zup = np.arange(z[0], z[-1] + dz, dz)
    nz = len(zup)

    K, G = velocity_to_modulus(Vp, Vs, Rho)

    C11 = np.zeros(nz)
    C33 = np.zeros(nz)
    C44 = np.zeros(nz)
    C66 = np.zeros(nz)
    C13 = np.zeros(nz)

    for k in range(nz):
        idx = np.where(np.abs(z - zup[k]) <= dz / 2)[0]
        if len(idx) == 0:
            # Fallback to nearest point if no points in window
            idx = [np.argmin(np.abs(z - zup[k]))]
        c11, c33, c44, c66, c12, c13 = modulus_to_stiffness(K[idx], G[idx])
        v = 1.0 / len(idx)
        A, C, D, M, _, F = backus_medium(c11, c33, c44, c66, c12, c13, v)
        C11[k] = A
        C33[k] = C
        C44[k] = D
        C66[k] = M
        C13[k] = F

    # Upscale density using moving average
    window_size = int(np.round(len(z) / nz))
    if window_size < 1:
        window_size = 1
    # Apply moving average by interpolation
    Rhoup_coarse = np.interp(zup, z, Rho)
    Rhoup = Rhoup_coarse.copy()

    # Interpolate back to original grid if method is not 'none'
    if interpmethod:
        interp_kwargs = {'kind': interpmethod, 'fill_value': 'extrapolate'}
        f_C11 = interp1d(zup, C11, **interp_kwargs)
        f_C33 = interp1d(zup, C33, **interp_kwargs)
        f_C44 = interp1d(zup, C44, **interp_kwargs)
        f_C66 = interp1d(zup, C66, **interp_kwargs)
        f_C13 = interp1d(zup, C13, **interp_kwargs)
        f_Rhoup = interp1d(zup, Rhoup, **interp_kwargs)

        C11 = f_C11(z)
        C33 = f_C33(z)
        C44 = f_C44(z)
        C66 = f_C66(z)
        C13 = f_C13(z)
        Rhoup = f_Rhoup(z)
        zup_out = z
    else:
        zup_out = zup

    epsilon, gamma, delta, Vp0up, Vs0up = stiffness_to_thomsen(C11, C33, C44, C66, C13, Rhoup)

    return Vp0up, Vs0up, Rhoup, epsilon, gamma, delta, zup_out

def backus_medium(C11, C33, C44, C66, C12, C13, v):
    """
    Backus Effective Voigt elements of a TI medium composed of several TI strata.

    Notes
    -----
    In a Backus medium, averaging is done over media and not samples.

    Parameters
    ----------
    C11, C33, C44, C66, C12, C13 : array_like
        Matrices of Voigt elements for each stratum, shape (Nsamples, Nstrata).
    v : array_like
        Thickness vector of each stratum, shape (Nsamples, Nstrata).

    Returns
    -------
    c11, c33, c44, c66, c12, c13 : ndarray
        Effective Voigt elements of the Backus-averaged medium, shape (Nsamples,).

    References
    ----------
    Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
    Cambridge University Press.
    """

    a = C11
    b = C12
    f = C13
    c = C33
    d = C44
    m = C66

    c33 = 1.0 / np.sum(v / c)
    c13 = c33 * np.sum(v * (f / c))
    c11 = np.sum(v * (a - f * f / c)) + c13
    c12 = np.sum(v * (b - f * f / c)) + c13
    c44 = 1.0 / np.sum(v / d)
    c66 = np.sum(v * m)

    return c11, c33, c44, c66, c12, c13

def backus_medium_velocity(Vp, Vs, Rho, v, theta):
    """
    Backus anisotropic velocities of a medium composed of homogeneous layers.


    Notes
    -----
    Averaging is done over media and not over samples.

    Parameters
    ----------
    Vp, Vs, Rho : array_like
        Matrices of P-wave velocity (m/s), S-wave velocity (m/s), and density (g/cc)
        for each medium, shape (Nsamples, Nmedia).
    v : array_like
        Matrix of volume fractions (or thickness weights) of each medium,
        shape (Nsamples, Nmedia).
    theta : float or array_like
        Angle between the TI 3-axis (vertical) and desired wave propagation direction,
        in degrees.

    Returns
    -------
    Vpeff : ndarray
        Effective P-wave velocity, shape (Nsamples,).
    Vsveff : ndarray
        Effective vertical S-wave (SV) velocity, shape (Nsamples,).
    Vsheff : ndarray
        Effective horizontal S-wave (SH) velocity, shape (Nsamples,).

    References
    ----------
    Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
    Cambridge University Press.

    Author
    ------
    Mostafa Abbasi, Ph.D., Geophysics
    email: abbasi.mstfa@gmail.com
    Last revision: 17-August-2022
    Copyright (c) 2022, Mostafa Abbasi
    """
    K, G = velocity_to_modulus(Vp, Vs, Rho)
    c11, c33, c44, c66, c12, c13 = modulus_to_stiffness(K, G)
    C11, C33, C44, C66, _, C13 = backus_medium(c11, c33, c44, c66, c12, c13, v)
    rho_eff = np.sum(Rho * v, axis=1)

    Vpeff, Vsveff, Vsheff = stiffness_to_velocity(C11, C33, C44, C66, C13, rho_eff, theta)

    return Vpeff, Vsveff, Vsheff

def backusupscale(z, Vp, Vs, Rho, znew):
    """
    Upscale elastic data into coarse scale by Backus averaging.


    Notes
    -----
    Midpoints of successive sample pairs in `znew` are selected as interval
    limits. Backus average of all elements in each interval is assigned to the
    corresponding element of `znew`.

    Parameters
    ----------
    z : array_like
        Input sample depths, shape (nsamples,).
    Vp : array_like
        P-wave velocity, shape (nsamples,).
    Vs : array_like
        S-wave velocity, shape (nsamples,).
    Rho : array_like
        Density (g/cc), shape (nsamples,).
    znew : array_like
        Output depths, shape (nnew,).

    Returns
    -------
    Vpup : ndarray
        Up-scaled P-velocity along symmetry axis, shape (nnew,).
    Vsup : ndarray
        Up-scaled S-velocity along symmetry axis, shape (nnew,).
    Rhoup : ndarray
        Up-scaled density, shape (nnew,).

    Author
    ------
    Mostafa Abbasi, Ph.D., Geophysics
    email: abbasi.mstfa@gmail.com
    Last revision: 17-August-2022
    Copyright (c) 2022, Mostafa Abbasi
    """

    if np.min(np.diff(z)) > np.min(np.diff(znew)):
        raise ValueError('New scale cannot be finer than original scale')

    dz = np.diff(znew)
    zint = np.concatenate(([znew[0]], znew[:-1] + dz / 2, [znew[-1]]))
    nz = len(znew)

    K, G = velocity_to_modulus(Vp, Vs, Rho)

    C11 = np.zeros(nz)
    C33 = np.zeros(nz)
    C44 = np.zeros(nz)
    C66 = np.zeros(nz)
    C13 = np.zeros(nz)
    Rhoup = np.zeros(nz)

    for k in range(nz):
        idx2 = np.where((z >= zint[k]) & (z < zint[k + 1]))[0]
        if len(idx2) == 0:
            # If no points in interval, use nearest neighbor
            idx2 = np.array([np.argmin(np.abs(z - znew[k]))])
        c11, c33, c44, c66, c12, c13 = modulus_to_stiffness(K[idx2], G[idx2])
        v = np.ones_like(c11) / len(idx2)
        A, C, D, M, _, F = backus_medium(c11, c33, c44, c66, c12, c13, v)
        C11[k] = A
        C33[k] = C
        C44[k] = D
        C66[k] = M
        C13[k] = F
        Rhoup[k] = np.mean(Rho[idx2])

    _, _, _, Vpup, Vsup = stiffness_to_thomsen(C11, C33, C44, C66, C13, Rhoup)

    return Vpup, Vsup, Rhoup
