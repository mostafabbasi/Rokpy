"""Module containing various AVO equations for PP reflectivity.

This module includes implementations of well-known AVO equations, both exact and
approximate, for calculating PP reflectivity based on rock properties and incident
angles."""

import numpy as np
from scipy.signal import convolve
from typing import Literal
from rokpy.backus import backus_avgerage
from rokpy.utilities import MultiEnum, condition_array

class AVOMethods:    
    """Class containing various AVO equations for PP reflectivity."""
    class AVOMethodName(MultiEnum):
        Zoeppritz       = 'zoeppritz'
        Bortfeld        = 'bortfeld'
        AkiRichards     = 'aki_richards'
        Shuey           = 'shuey'
        SmithGidlow     = 'smith_gidlow'
        Fatti           = 'fatti'
        Goodway         = 'goodway'
        GrayGoodway     = 'gray_goodway'
        Hilterman       = 'hilterman'
        QuadraticZoeppritz = 'quadratic_zoepritz'
        Wiggins         = 'wiggins'
        LogAkiRichards  = 'aki_richards_log'

    def zoeppritz(Vp, Vs, Rho, theta):
        """
        Zoeppritz's (1919) exact equation for PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Zhi L, Chen S, Li XY. Joint AVO inversion of PP and PS waves using exact
        Zoeppritz equation. InSEG Technical Program Expanded Abstracts 2013
        Sep (pp. 457-461). Society of Exploration Geophysicists.
        """

        # Avoid division by zero at 0 degrees
        theta = np.where(theta == 0, np.finfo(float).eps, theta)

        Vp = np.append(Vp, Vp[-1])
        Vs = np.append(Vs, Vs[-1])
        Rho = np.append(Rho, Rho[-1])

        vp1 = Vp[:-1]
        vs1 = Vs[:-1]
        rho1 = Rho[:-1]
        vp2 = Vp[1:]
        vs2 = Vs[1:]
        rho2 = Rho[1:]

        r1 = vp2[:, None] / vp1[:, None]
        r2 = vs1[:, None] / vp1[:, None]
        r3 = vs2[:, None] / vp1[:, None]
        r4 = rho2[:, None] / rho1[:, None]

        sin_theta = np.sin(np.deg2rad(theta))
        q1 = np.sqrt(1 - sin_theta**2)
        q2 = np.sqrt(np.maximum(0, 1 - (r1 * sin_theta)**2))
        q3 = np.sqrt(np.maximum(0, 1 - (r2 * sin_theta)**2))
        q4 = np.sqrt(np.maximum(0, 1 - (r3 * sin_theta)**2))

        T1 = sin_theta / q1
        T2 = r1 * sin_theta / q2
        T3 = r2 * sin_theta / q3
        T4 = r3 * sin_theta / q4

        Q = 2 * sin_theta**2 * (r4 * r3 * r3 - r2 * r2)
        a = r4 - Q - 1
        b = r4 - Q
        c = 1 + Q

        numerator = (Q * Q - r4 * T1 * T4 + r4 * T2 * T3 - c * c * T1 * T3 +
                    b * b * T2 * T4 + a * a * T1 * T2 * T3 * T4)
        denominator = (Q * Q + r4 * T1 * T4 + r4 * T2 * T3 + c * c * T1 * T3 +
                    b * b * T2 * T4 + a * a * T1 * T2 * T3 * T4)

        rpp = numerator / denominator

        return rpp

    def bortfeld(Vp, Vs, Rho, theta):
        """
        Bortfeld approximation to Zoeppritz PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        # Avoid division by zero at 0 degrees
        theta = np.where(theta == 0, np.finfo(float).eps, theta)

        Vp = np.append(Vp, Vp[-1])
        Vs = np.append(Vs, Vs[-1])
        Rho = np.append(Rho, Rho[-1])

        vp1 = Vp[:-1]
        vs1 = Vs[:-1]
        rho1 = Rho[:-1]
        vp2 = Vp[1:]
        vs2 = Vs[1:]
        rho2 = Rho[1:]

        sin_theta = np.sin(np.deg2rad(theta))
        theta2_rad = np.arcsin(np.clip((vp2[:, None] / vp1[:, None]) * sin_theta, -1, 1))
        cos_theta = np.cos(np.deg2rad(theta))
        cos_theta2 = np.cos(theta2_rad)

        c = cos_theta / cos_theta2

        term1 = 0.5 * np.log(vp2[:, None] / vp1[:, None] * rho2[:, None] / rho1[:, None] * c)
        term2 = (vs1[:, None]**2 - vs2[:, None]**2) * (sin_theta / vp1[:, None])**2

        # Handle log(vs2./vs1) safely
        log_vs_ratio = np.log(vs2[:, None] / vs1[:, None])
        # Avoid division by zero; where log_vs_ratio is zero, set ratio to zero
        ratio = np.zeros_like(log_vs_ratio)
        nonzero = np.abs(log_vs_ratio) > np.finfo(float).eps
        ratio[nonzero] = np.log(rho2[:, None] / rho1[:, None])[nonzero] / log_vs_ratio[nonzero]

        term2 = term2 * (2 + ratio)

        rpp = term1 + term2
        rpp = np.nan_to_num(rpp, nan=0.0)

        return rpp

    def aki_richards(Vp, Vs, Rho, theta):
        """
        Aki-Richards weak approximation of PP reflectivity.
        
        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Buland, A. and Omre, H., 2003. Bayesian linearized AVO inversion.
        Geophysics, 68(1), pp.185-198.
        """

        nt = Vp.shape[0]

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)
        RRho = (D @ Rho) / (M @ Rho)

        cp, cs, cr = AVOMethods._aki_richards_coefs(theta, vpvs)

        rpp = cp * RVp[:, None] + cs * RVs[:, None] + cr * RRho[:, None]

        return rpp

    def shuey(Vp, Vs, Rho, theta, terms:Literal['two','three']='three'):
        """
        Shuey's weak approximation of PP reflectivity.

        Notes
        -----
        - If `terms=3`, this equation returns the same results as Aki-Richards.
        - If `terms=2` and `vpvs=2`, this equation returns the same results as
        Wiggins and Hilterman.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees
        terms : int
            Number of terms in the approximation (2 or 3).

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)
        RRho = (D @ Rho) / (M @ Rho)

        Rp0 = 0.5 * (RVp + RRho)
        H = 0.5 * RVp - 2 * vsvp2 * (2 * RVs + RRho)
        C = 0.5 * RVp

        theta_rad = np.deg2rad(theta)
        sin_theta2 = np.sin(theta_rad) ** 2

        if terms == 'three':
            tan_theta2 = np.tan(theta_rad) ** 2
            rpp = Rp0[:, None] + H[:, None] * sin_theta2 + C[:, None] * (tan_theta2 - sin_theta2)
        elif terms == 'two':
            rpp = Rp0[:, None] + H[:, None] * sin_theta2
        else:
            raise ValueError("terms must be 2 or 3")

        return rpp

    def smith_gidlow(Vp, Vs, theta):
        """
        Smith and Gidlow's (1985) weak approximation of PP reflectivity.

        Notes
        -----
        This relation assumes rho = a * Vp^0.25.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)

        theta_rad = np.deg2rad(theta)
        sin_theta2 = np.sin(theta_rad) ** 2
        tan_theta2 = np.tan(theta_rad) ** 2

        c = 5/8 - 0.5 * vsvp2 * sin_theta2 + 0.5 * tan_theta2
        d = -4 * vsvp2 * sin_theta2

        rpp = c * RVp[:, None] + d * RVs[:, None]

        return rpp

    def fatti(Vp, Vs, Rho, theta, terms:Literal['two','three']='three'):
        """
        Fatti et al's (1994) weak approximation of PP reflectivity.

        Notes
        -----
        - If `terms=3`, this equation returns the same results as Aki-Richards.
        - This equation is based on Smith and Gidlow (1986).

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees
        terms : int
            Number of terms in the approximation (2 or 3).

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)
        RRho = (D @ Rho) / (M @ Rho)

        Rp = 0.5 * (RVp + RRho)
        Rs = 0.5 * (RVs + RRho)

        theta_rad = np.deg2rad(theta)
        tan_theta2 = np.tan(theta_rad) ** 2
        sin_theta2 = np.sin(theta_rad) ** 2

        cp = 1 + tan_theta2
        cs = -8 * vsvp2 * sin_theta2
        cr = 2 * vsvp2 * sin_theta2 - 0.5 * tan_theta2

        if terms.lower() == 'three':
            rpp = cp * Rp[:, None] + cs * Rs[:, None] + cr * RRho[:, None]
        elif terms == 'two':
            rpp = cp * Rp[:, None] + cs * Rs[:, None]
        else:
            raise ValueError("terms must be 2 or 3")

        return rpp

    def goodway(Vp, Vs, Rho, theta):
        """
        Goodway's (1998) weak approximation of PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Buland, A. and Omre, H., 2003. Bayesian linearized AVO inversion.
        Geophysics, 68(1), pp.185-198.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = Vp.shape[0]

        Ip = Vp*Rho
        Is = Vs*Rho

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RIp = (D @ Ip) / (M @ Ip)
        RIs = (D @ Is) / (M @ Is)
        RRho = (D @ Rho) / (M @ Rho)

        theta_rad = np.deg2rad(theta)
        tan_theta2 = np.tan(theta_rad) ** 2
        sin_theta2 = np.sin(theta_rad) ** 2

        cp = 0.5 * (1 + tan_theta2)
        cs = -4 * vsvp2[:, None] * sin_theta2
        cr = -(0.5 * tan_theta2 - 2 * vsvp2[:, None] * sin_theta2)

        rpp = cp * RIp[:, None] + cs * RIs[:, None] + cr * RRho[:, None]

        return rpp

    def gray_goodway(Vp, Vs, Rho, theta):
        """
        Gray et al's (1999) weak approximation of PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        G = Vs**2 * Rho
        K = Vp**2 *Rho - 4/3*G

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RK = (D @ K) / (M @ K)
        RG = (D @ G) / (M @ G)
        RRho = (D @ Rho) / (M @ Rho)

        theta_rad = np.deg2rad(theta)
        sec_theta2 = 1.0 / np.cos(theta_rad) ** 2
        sin_theta2 = np.sin(theta_rad) ** 2

        A = (1/4 - (1/3) * vsvp2[:, None]) * RK[:, None]
        B = vsvp2[:, None] * RG[:, None]
        C = RRho[:, None]

        rpp = (A * sec_theta2 +
            B * (0.5 * sec_theta2 - 2 * sin_theta2) +
            C * (0.5 - 0.25 * sec_theta2))

        return rpp

    def hilterman(Vp, Vs, Rho, theta):
        """
        Hilterman's (1987) weak approximation of PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        Notes
        -----
        - This modified form interprets near-offset traces as P-wave impedance
        and intermediate-offset traces as Poisson ratio contrasts (Castagna, 1993).
        - Equivalent to the two-term Shuey approximation when Vp/Vs = 2.
        - Derived from Shuey's equation by:
            1. Keeping only the first two terms.
            2. Assuming nu = 1/3 (i.e., Vp/Vs = 2).        

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """
        velocity_ratio = Vp/Vs
        nu = ((velocity_ratio**2 - 2) / (2*velocity_ratio**2 - 2))
        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        RVp = (D @ Vp) / (M @ Vp)
        RRho = (D @ Rho) / (M @ Rho)

        Rp = 0.5 * (RVp + RRho)
        PR = (D @ nu) / (1 - M @ nu) ** 2

        theta_rad = np.deg2rad(theta)
        cos_theta2 = np.cos(theta_rad) ** 2
        sin_theta2 = np.sin(theta_rad) ** 2

        rpp = Rp[:, None] * cos_theta2 + PR[:, None] * sin_theta2

        return rpp

    def quadratic_zoeppritz(Vp, Vs, Rho, theta):
        """
        Quadratic approximation to Zoeppritz PP reflectivity.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Wang, Y., 1999. Approximations to the Zoeppritz equations and their use
        in AVO analysis. Geophysics, 64(6), pp.1920-1927.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = Vp.shape[0]

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        vsvp2 = vpvs ** (-2)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)
        RRho = (D @ Rho) / (M @ Rho)
        Rq = RRho + 2 * RVs

        theta_rad = np.deg2rad(theta)
        tan_theta2 = np.tan(theta_rad) ** 2
        sin_theta2 = np.sin(theta_rad) ** 2
        cos_theta = np.cos(theta_rad)

        cp = 0.5 * (1 + tan_theta2)
        cs = -4 * vsvp2[:, None] * sin_theta2
        cr = 0.5 - 2 * vsvp2[:, None] * sin_theta2
        cq = (vsvp2 ** (3/2))[:, None] * cos_theta * sin_theta2

        rpp = (cp * RVp[:, None] +
            cs * RVs[:, None] +
            cr * RRho[:, None] +
            cq * (Rq[:, None] ** 2))

        return rpp

    def wiggins(Vp, Vs, Rho, theta):
        """
        Wiggins et al's (1983) weak approximation of PP reflectivity.
        This equation was further published by Gelfand and Larner (1986).

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        Notes
        -----
        - Assumes Vp/Vs = 2.
        - Equivalent to the two-term Shuey approximation when Vp/Vs = 2.
        - The Hilterman model simplifies Shuey's equation by:
            1. Using only the first two terms.
            2. Setting Vp/Vs = 2.

        References
        ----------
        Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook.
        Cambridge University Press.
        """

        theta = AVOMethods._avgppangle(theta, Vp)
        nt = len(Vp)

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        RVp = (D @ Vp) / (M @ Vp)
        RVs = (D @ Vs) / (M @ Vs)
        RRho = (D @ Rho) / (M @ Rho)

        Rp = 0.5 * (RVp + RRho)
        Rs = 0.5 * (RVs + RRho)
        G = Rp - 2 * Rs

        theta_rad = np.deg2rad(theta)
        sin_theta2 = np.sin(theta_rad) ** 2

        rpp = Rp[:, None] + G[:, None] * sin_theta2

        return rpp

    def aki_richards_log(Vp, Vs, Rho, theta):
        """
        Aki-Richards weak approximation of PP reflectivity using logarithmic contrasts.

        Parameters
        ----------
        Vp : np.ndarray
            P-wave velocity vector
        Vs : np.ndarray
            S-wave velocity vector
        Rho : np.ndarray
            Density vector
        theta : np.ndarray
            Incident angles in degrees

        Returns
        -------
        rpp : ndarray
            PP reflectivities.

        References
        ----------
        Buland, A. and Omre, H., 2003. Bayesian linearized AVO inversion.
        Geophysics, 68(1), pp.185-198.
        """

        nt = Vp.shape[0]

        M = AVOMethods._avgmat(nt)
        D = AVOMethods._diffmat(nt)

        vpvs = (M @ Vp) / (M @ Vs)
        RVp = D @ np.log(Vp)
        RVs = D @ np.log(Vs)
        RRho = D @ np.log(Rho)

        cp, cs, cr = AVOMethods._aki_richards_coefs(theta, vpvs)

        rpp = cp * RVp[:, None] + cs * RVs[:, None] + cr * RRho[:, None]

        return rpp

    def _avgmat(nt):
        """
        Returns the size-preserving two-point averaging matrix.
        
        Parameters
        ----------
        nt : int
            Number of samples in the input vector.

        Returns
        -------
        M : ndarray
            Averaging matrix of shape (nt - 1, nt) such that M @ v computes
            the two-point average: 0.5 * (v[i] + v[i+1]) for i = 0,...,nt-2.
        """

        M = 0.5 * (np.eye(nt) + np.diag(np.ones(nt - 1), k=1))
        M[-1,:] = M[-2,:]  # Handle last row to preserve size
        return M

    def _diffmat(nt):
        """
        Returns the size-preserving differentiating matrix.
        
        Parameters
        ----------
        nt : int
            Number of samples in the input vector.

        Returns
        -------
        D : ndarray
            Differentiating matrix of shape (nt - 1, nt) such that D @ v computes
            the first-order forward difference: v[i+1] - v[i] for i = 0,...,nt-2.
        """

        D = -np.eye(nt) + np.diag(np.ones(nt - 1), k=1)
        D[-1,:] = D[-2,:]  # Handle last row to preserve size
        return D

    def _aki_richards_coefs(theta, vpvs):
        """
        Coefficients of Aki-Richards weak PP reflectivity.
        
        Parameters
        ----------
        theta : np.ndarray
            Incident angles in degrees
        vpvs : np.ndarray
            Average P- to S-velocity ratio (N,).

        Returns
        -------
        cp : ndarray
            P-wave coefficient (N, Ntheta).
        cs : ndarray
            S-wave coefficient (N, Ntheta).
        cr : ndarray
            Density coefficient (N, Ntheta).

        References
        ----------
        Buland, A. and Omre, H., 2003. Bayesian linearized AVO inversion.
        Geophysics, 68(1), pp.185-198.
        """

        theta = np.asarray(theta)
        vpvs = np.asarray(vpvs)

        vsvp2 = vpvs[:, None] ** (-2)
        theta_rad = np.deg2rad(theta)

        cp = 0.5 * (1 + np.tan(theta_rad) ** 2) * np.ones_like(vsvp2)
        cs = -4 * vsvp2 * np.sin(theta_rad) ** 2
        cr = 0.5 * (1 - 4 * vsvp2 * np.sin(theta_rad) ** 2)

        return cp, cs, cr

    def _avgppangle(theta1, Vp):
        """
        Returns average of incident and transmitted P-wave angles.

        Notes
        -----
        This function uses Snell's law: sin(theta1) / Vp1 = sin(theta2) / Vp2.

        Parameters
        ----------
        theta1 : np.ndarray
            Incident P-wave angles in degrees
        Vp : np.ndarray
            P-wave velocity vector

        Returns
        -------
        thetahat : ndarray
            Average angle (thetahat = (theta1 + theta2) / 2).

        References
        ----------
        Innanen, K., 2011. Hidden nonlinearities in the Aki-Richards approximation.
        CREWES Annual Research Report, University of Calgary.
        """

        Vp2 = np.roll(Vp, -1)  # circshift(Vp, -1) and remove last
        Vp1 = Vp

        sin_theta1 = np.sin(np.deg2rad(theta1))
        sin_theta2 = (Vp2[:, None] / Vp1[:, None]) * sin_theta1
        # Clip to [-1, 1] to avoid invalid arcsin due to numerical errors
        sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
        theta2 = np.rad2deg(np.arcsin(sin_theta2))

        thetahat = 0.5 * (theta1 + theta2)
        return thetahat

class AVOModel:
    def __init__(self, theta, wavelet=None, dt=None, method_name:AVOMethods.AVOMethodName | Literal['zoeppritz','bortfeld','aki_richards','shuey','smith_gidlow','fatti','goodway','gray_goodway','hilterman','quadratic_zoepritz','wiggins','aki_richards_log'] = 'aki_richards'):
        self.theta = theta
        self.method_name = method_name
        self.wavelet = wavelet
        self.dt = dt

    @property
    def method(self) -> callable:
        """Return the callable implementation for the selected method.

        The returned function is looked up on the `method_class` using the
        selected `method_name`.
        """
        return getattr(AVOMethods, self.method_name)

    def reflectivity(self, Vp, Vs, Rho):
        Vp = condition_array(Vp)
        Vs = condition_array(Vs)
        Rho = condition_array(Rho)
        avo_method = getattr(AVOMethods, self.method_name)
        return avo_method(Vp, Vs, Rho, self.theta)
    
    def seismic (self, Vp, Vs, Rho, time):
        Vpup, Vsup, Rhoup,_, _, _, time_regular = backus_avgerage(Vp, Vs, Rho, time, self.dt, interpmethod=None)
        rpp = self.reflectivity(Vpup, Vsup, Rhoup)
        seis = convolution_seis(rpp, self.wavelet)
        return seis, rpp, time_regular

def convolution_seis(rpp, wavelet, dt=0.002):
    """
    Seismic convolutional forward model.

    Parameters
    ----------
    rpp : np.ndarray
        PP reflectivity matrix (nt, ntheta).
    wavelet : np.ndarray
        Wavelet matrix (ntw, ntheta) or (ntw,) for a single wavelet.
    dt : float, optional
        Sample rate in seconds. If not provided, defaults to 0.002.
        If `dt` is given in milliseconds, amplitudes are scaled by 0.001.

    Returns
    -------
    seis : ndarray
        Seismic angle gather (nt, ntheta).

    References
    ----------
    Buland, A. and Omre, H., 2003. Bayesian linearized AVO inversion.
    Geophysics, 68(1), pp.185-198.
    """

    ntheta = rpp.shape[1]
    if wavelet.ndim == 1:
        wavelet = np.tile(wavelet[:, None], (1, ntheta))
    else:
        assert wavelet.shape[1] == ntheta, "Number of wavelets must match number of angles."

    seis = np.zeros_like(rpp)
    for i in range(ntheta):
        conv_result = convolve(rpp[:,i], wavelet[:, i], mode='same')
        seis[:, i] = (1.0 / dt) * conv_result

    return seis

def ricker(dt, fdom, length):
    """
    Creates a zero-centered Ricker wavelet.
    
    Parameters
    ----------
    dt : float
        Sample rate in milliseconds.
    fdom : float
        Center frequency of the wavelet (Hz).
    length : float
        Total length of the wavelet in milliseconds.

    Returns
    -------
    w : ndarray
        Ricker wavelet (N,).
    tw : ndarray
        Time vector corresponding to the wavelet (N,).
    """

    # Create time vector symmetric around zero
    t_left = np.arange(-length / 2, 0 + dt, dt)  # include 0
    t_right = np.arange(dt, length / 2 + dt, dt)
    tw = np.concatenate((t_left, t_right))

    beta = (tw * (fdom * np.pi)) ** 2
    w = (1 - 2 * beta) * np.exp(-beta)

    return w, tw

