"""
Rock physics extra tools and utility functions

This module provides essential utility functions for rock physics computations,
including array handling, well log management, microstructural parameters, and
poroelastic/Biot theory calculations. It serves as the computational toolkit
for the rock physics package.
"""

from dataclasses import dataclass, field
import random
from typing import Literal
import numpy as np
from .conversions import modulus_to_poisson
from enum import Enum
from .constants import PropertyTemplate, PropertyTemplates
from functools import wraps
from scipy.ndimage import uniform_filter1d, median_filter


def normalizearrays(func):
    """
    Decorator that:
    Converts all array-like inputs to at least 1D np.ndarray via rparray.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        norm_args = [np.atleast_1d(np.asarray(a, dtype=float)) for a in args]
        norm_kwargs = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in kwargs.items()}
        return func(*norm_args, **norm_kwargs)
    return wrapper

def is_array_like(x):
    """Return True for lists, tuples, or numpy arrays (but not strings)."""
    return isinstance(x, (list, tuple, np.ndarray))

class MultiEnum(str, Enum):
    def __new__(cls, *values):
        # The first value is the canonical string value
        obj = str.__new__(cls, values[0])
        obj._value_ = values[0]
        obj._aliases_ = set(values)
        return obj

    @classmethod
    def _missing_(cls, value):
        """Called when Enum(value) fails — try to match aliases."""
        for member in cls:
            if value in member._aliases_:
                return member
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")

    def __eq__(self, other):
        """Allow comparison with any alias string."""
        if isinstance(other.upper(), str):
            return other in self._aliases_
        return super().__eq__(other.upper())

    def __hash__(self):
        return hash(self._value_)

    def aliases(self):
        """Return all aliases of this enum member."""
        return self._aliases_

def condition_array(arr, median_window=3):
    if median_window:
        arr = moving_median(arr, window_size=median_window)
    ok_mask = ~np.isnan(arr)
    xp = ok_mask.ravel().nonzero()[0]
    fp = arr[ok_mask]
    x  = np.isnan(arr).ravel().nonzero()[0]
    arr[np.isnan(arr)] = np.interp(x, xp, fp)
    return arr

def rparray(arr):
    """rparray indices on the first axis (axis=0) refer elements and indices on the next axes refer to samples
    """
    arr = np.atleast_1d(np.asarray(arr, dtype=float))
    return arr

def moving_average(data, window_size=3):
    """
    Apply a moving average filter to a 1D numpy array.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input 1D array
    window_size : int
        Number of samples in the moving window (must be positive odd integer)
    
    Returns:
    --------
    numpy.ndarray
        Filtered array with same shape as input
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd for symmetric filtering")
    
    return uniform_filter1d(data, size=window_size, mode='nearest')

def moving_median(data, window_size=3):
    """
    Apply a moving median filter to a 1D numpy array.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input 1D array
    window_size : int
        Number of samples in the moving window (must be positive odd integer)
    
    Returns:
    --------
    numpy.ndarray
        Filtered array with same shape as input
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd for symmetric filtering")
    
    return median_filter(data, size=window_size, mode='nearest')

def crack_density(porosity: float | np.ndarray, 
                  aspect_ratio: float | np.ndarray):
    """
    Estimate Crack density or number of cracks per unit volume times the crack radius cubed.

    Parameters
    ----------
    porosity : float or ndarray
        Total porosity (fraction, 0-1).
    aspect_ratio : float or ndarray
        Crack aspect ratio (a/R), dimensionless (small number << 1).

    Returns
    -------
    crack_density : float or ndarray
        Crack density parameter (dimensionless). Uses the
        approximation rho_crack = (3 / (4*pi)) * porosity / aspect_ratio.

    Notes
    -----
    Crack density here follows the common seismological/geomechanical
    definition proportional to crack volume divided by aspect ratio.
    """
    return (3/4/np.pi) * porosity/aspect_ratio

def contact_ratio(pressure: float | np.ndarray, 
                  bulk: float | np.ndarray, 
                  shear: float | np.ndarray, 
                  porosity: float | np.ndarray, 
                  contact_no: float | np.ndarray):
    """
    Compute Relative contact radius of random identical spheres under a hydrostatic confining pressure.

    Parameters
    ----------
    pressure : float or ndarray
        Hydrostatic confining pressure (same units as used consistently below;
        see Notes). Typical input in Pascals (Pa) or MPa — user must be consistent.
    bulk : float or ndarray
        Bulk modulus of the granular frame (GPa).
    shear : float or ndarray
        Shear modulus of the granular frame (GPa).
    porosity : float or ndarray
        Porosity (fraction, 0-1).
    contact_no : float or ndarray
        Coordination number (average number of contacts per grain), dimensionless.

    Returns
    -------
    a2R : float or ndarray
        Contact-size parameter a^2/R raised to the 1/3 power (dimension: depends on
        unit choices; treated here as dimensionless after consistent unit conversion).

    Notes
    -----
    Uses an approximate Hertz-Mindlin contact scaling:
    a2R = [ (3*pi/2) * (1 - nu) / (Z * (1 - phi) * G) * pressure ]^(1/3),
    where nu is Poisson's ratio computed from bulk and shear moduli. The function
    expects bulk and shear in GPa; no internal unit conversion for pressure is
    performed — keep pressure consistent with modulus units (e.g., Pa with GPa
    converted externally or pressure in GPa).
    """
    poisson = modulus_to_poisson(bulk, shear)
    a2R = (3*np.pi/2 * (1-poisson)/(contact_no*(1-porosity)*shear) *pressure)**(1/3)
    return a2R

def relative_stiffness(bulk: float | np.ndarray, 
                       shear: float | np.ndarray, 
                       a2R: float | np.ndarray, 
                       adhesion_coef: float = 1) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Estimate ratio of normal (compressional) and tangential (shear) stiffnesses to grain radius.

    Parameters
    ----------
    bulk : float or ndarray
        Bulk modulus of the frame (GPa).
    shear : float or ndarray
        Shear modulus of the frame (GPa).
    a2R : float or ndarray
        Ratio of Grain contact radius to Grain radius
        This parameter may be produced by contact_ratio (see `utilities.contact_ratio`).
    adhesion_coef : float, optional
        Adhesion coefficient for tangential stiffness (dimensionless). Default is 1.

    Returns
    -------
    Sn2R : float or ndarray
        Ratio of normal stiffness to grain radius 
    St2R : float or ndarray
        Ratio of tangential stiffness to grain radius

    Notes
    -----
    Uses simple scaling relations:
      Sn2R = 4 * G * a2R / (1 - nu)
      St2R = adhesion_coef * 8 * G * a2R / (2 - nu)
    where nu is Poisson's ratio computed from bulk and shear moduli.
    """
    poisson = modulus_to_poisson(bulk, shear)
    Sn2R = 4*shear*a2R/(1-poisson)
    St2R = adhesion_coef*8*shear*a2R/(2-poisson)
    return Sn2R, St2R

def coordination_no (porosity , friction):
    """
    Estimate coordination number (average contacts per grain) as a function of porosity.

    Parameters
    ----------
    porosity : float or ndarray
        Porosity (fraction, 0-1).
    friction : {0, 1}
        Friction flag: 
        1 = with friction, General model  by Garcia and Medina (2006), 
        0 = frictionless model  by Makse et al. (2004)
    Returns
    -------
    C : float or ndarray
        Estimated coordination number (dimensionless).

    Raises
    ------
    ValueError
        If porosity is greater than the empirical upper limit for the chosen friction
        case or if an unsupported friction value is passed.

    """
    if friction == 1:
        C0 = 4.46
        phi0 = 0.384
        if porosity<phi0:
            C = C0 + 9.7*(phi0-porosity)**0.48
        else:
            raise(ValueError('Porosity must be less than {:.3f}'.format(phi0)))
    elif friction == 0:
        C0 = 6
        phi0 = 0.37
        if porosity<phi0:
            C = C0 + 9.1*(phi0-porosity)**0.48
        else:
            raise(ValueError('Porosity must be less than {:.3f}'.format(phi0)))
    else:
        raise(ValueError('friction must be 0 or 1'))
    return C

def tortuosity(porosity: float | np.ndarray, 
               r: float = 3) -> float | np.ndarray:
    """
    Tortuosity (also known as structure factor) of the pores based on Berryman's (1981) relation.

    Parameters
    ----------
    porosity : float or ndarray
        Porosity (fraction, 0-1).
    r : float, optional
        Empirical shape factor (dimensionless). Default is 3.
        r = 1/2     for spheres
        r = 1       for uniform cylindrical pores
        r = 3       for random system of pores (Stoll, 1977)

    Returns
    -------
    tau : float or ndarray
        Estimated tortuosity (dimensionless). Uses tau = 1 - r*(1 - 1/phi).

    Notes
    -----
    This is a very simple, potentially singular model for small porosity. Use with care.
    """
    return 1 - r*(1 - 1/porosity)

def biot_coef( dry_bulk: float | np.ndarray, 
               mineralset_bulk: float | np.ndarray ) -> float | np.ndarray:
    """
    Compute Biot's effective stress coefficient (alpha).

    Parameters
    ----------
    dry_bulk : float or ndarray
        Dry-rock (frame) bulk modulus (GPa).
    mineralset_bulk : float or ndarray
        Bulk modulus of the mineral/end-member solid matrix (GPa).

    Returns
    -------
    beta : float or ndarray
        Biot coefficient (dimensionless), beta = 1 - K_dry / K_min.

    Notes
    -----
    Values range from 0 (rigid solid) to ~1 (very compliant frame).
    """
    beta = 1- dry_bulk/mineralset_bulk
    return beta

def biot_coef_krief(porosity: float | np.ndarray) -> float | np.ndarray:
    """
    Krief-style estimate of Biot coefficient using an empirical porosity relation.

    Parameters
    ----------
    porosity : float or ndarray
        Porosity (fraction, 0-1).

    Returns
    -------
    beta : float or ndarray
        Approximate Biot coefficient (dimensionless) using beta = 1 - (1-phi)^m,
        where m = 3/(1-phi).

    Notes
    -----
    This is an empirical approximation; valid for typical sedimentary porosities.
    """
    m = 3/(1-porosity)
    beta = 1 - (1-porosity)**m
    return beta

def reuss_intercept_porosity( saturated_bulk: float | np.ndarray, 
                              mineralset_bulk: float | np.ndarray, 
                              fluidset_bulk: float | np.ndarray, 
                              porosity: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute the Reuss-bound intercept porosity and the corresponding Reuss bulk modulus.

    Parameters
    ----------
    saturated_bulk : float or ndarray
        Bulk modulus of the saturated rock (GPa).
    mineralset_bulk : float or ndarray
        Bulk modulus of the mineral solid mix (GPa).
    fluidset_bulk : float or ndarray
        Bulk modulus of the pore fluid mixture (GPa).
    porosity : float or ndarray
        Porosity (fraction, 0-1).

    Returns
    -------
    phiR : float or ndarray
        Intercept porosity at which a Reuss-type bounding mixture is implied (fraction).
    KR : float or ndarray
        Reuss bound bulk modulus at phiR (GPa).

    Notes
    -----
    The function calls BoundMethods.reuss from the models module to compute the Reuss
    bound for a two-component mixture given fractions [1-phiR, phiR].
    """
    from .models import BoundMethods
    phiR = ((mineralset_bulk-saturated_bulk)/(mineralset_bulk*porosity) + (fluidset_bulk-mineralset_bulk)/fluidset_bulk ) / ((saturated_bulk-mineralset_bulk)*(mineralset_bulk-fluidset_bulk)/(porosity*mineralset_bulk*fluidset_bulk))
    KR = BoundMethods.reuss([mineralset_bulk,fluidset_bulk] , [1-phiR,phiR])
    return phiR, KR

def biot_characteristic_frequency( porosity: float | np.ndarray, 
                                   fluidset_density: float | np.ndarray, 
                                   viscosity_to_permeability: float | np.ndarray) -> float | np.ndarray:
    """
    Estimate Biot's characteristic (viscous) frequency for squirt/poroelastic effects.

    Parameters
    ----------
    porosity : float or ndarray
        Porosity (fraction, 0-1).
    fluidset_density : float or ndarray
        Fluid bulk density in grams per cubic centimeter (g/cc).
    viscosity_to_permeability : float or ndarray
        Ratio of viscosity (cp) to permeability (md). 

    Returns
    -------
    fc : float or ndarray
        Characteristic frequency in Hz (1/s).

    Notes
    -----
    The implementation converts fluid density from g/cc to kg/m^3 by multiplying
    by 1000. The viscosity/permeability ratio is scaled by (1e-3/1e-15) to map
    common field units (centipoise and millidarcy) to Pa·s / m^2; ensure inputs are
    consistent with this assumption.
    """
    fluidset_density = fluidset_density*1000
    viscosity_to_permeability = (1e-3/1e-15)*viscosity_to_permeability #Convert to Metric
    fc = 1/(2*np.pi) * porosity * viscosity_to_permeability / fluidset_density
    return fc


@dataclass
class Well:
    name: str
    coordinates: tuple[float]
    KB: float = 0
    TD: float = 4000
    SE: float = 0

@dataclass
class WellLog:
    logname: str
    depth: np.ndarray
    data: np.ndarray
    wellname: Well | str
    template: PropertyTemplate =  field(default_factory = lambda: PropertyTemplates().General)
    color: tuple = field(default=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    
    def __post_init__(self):
        self.id = random.getrandbits(128)

    @property
    def min(self):
        return self.data.min()

    @property
    def max(self):
        return self.data.max()

    @property
    def mean(self):
        return self.data.mean()
    
    @property
    def std(self):
        return self.data.std()
