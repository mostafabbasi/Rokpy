"""
Inclusion-based relations of effective medium theory

Effective medium methods based on inclusion theory and microstructural modeling.

This module implements rock physics models that treat porous rocks as a background
matrix containing inclusions (spherical, ellipsoidal, or crack-like features) with
different elastic properties. Inclusion models account for pore geometry, shape,
and aspect ratios to predict effective elastic moduli from microstructural parameters.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple
import warnings
import numpy as np
import random

from rokpy import conversions
from rokpy.utilities import MultiEnum, rparray
from rokpy.constants import ElasticPropertySet, FluidsPropertyTable
from rokpy.effective_medium.bound_methods import BoundMethods
if TYPE_CHECKING:
    from rokpy.materials import Material


class ShapeName(MultiEnum):
    SPHERE   = 'SPHERE', 'O'
    NEEDLE   = 'NEEDLE', '|'
    DISK     = 'DISK', '-', '_'
    CRACK    = 'CRACK', '='
    SPHEROID = 'SPHEROID', '0'

@dataclass
class Inclusion:
    """Inclusion class representing a single inclusion in a host material.
    
    Parameters
    ----------
    shape_name : Literal['SPHEROID','SPHERE', 'NEEDLE', 'DISK', 'CRACK']
        Shape of the inclusion.
    aspect_ratio : np.ndarray
        Aspect ratio(s) of the inclusion.
    content : ElasticPropertySet
        Elastic properties of the inclusion.
    host : Material
        Host material containing the inclusion.
    fraction_in_host : float
        Fraction of the host material occupied by this inclusion.
        This property control what fraction of host porosity belong to this inclusion.
        If a single inclusion is present, this value should be 1.
        If multiple inclusions are present, their fractions should sum to 1.
    isstiff : bool
        Flag indicating if the inclusion is A stiff or compliant/crack crack.
        non-stiff inclusions are closed under high pressure"""
    shape_name: Literal['SPHEROID','SPHERE', 'NEEDLE', 'DISK', 'CRACK']
    aspect_ratio: np.ndarray
    content: ElasticPropertySet = field(default_factory=lambda: FluidsPropertyTable().Dry)
    host: "Material" = None
    fraction_in_host: float = 1.0
    isstiff: bool = False

    def __post_init__(self):
        self.id = random.getrandbits(128)

    @property
    def bulk(self):
        return self.content.bulk

    @property
    def shear(self):
        return self.content.shear

    @property
    def type(self):
        return self.host.type

    def __hash__(self):
        return hash(self.id)
    
    def __str__(self):
        return '(Shape={}, Content={}, aspect(mean)={:.2f}, Host={})'.format(\
                self.shape_name, \
                self.content.type, \
                np.mean(self.aspect_ratio), \
                str(self.host.type))
    
    def __repr__(self):
        return self.__str__()

class InclusionMethods():
    class InclusionMethodName(MultiEnum):
        MoriTanaka = 'mori_tanaka', 'mt'
        KusterToksoz = 'kuster_toksoz', 'kt'
        SelfConsistent = 'self_consistent', 'sc'
        DEM = 'dem', 'differential_effective_medium'
        ModifiedDEM = 'dem_modified', 'mdem'
        
    @staticmethod
    def eshelby_factors(host_bulk : np.ndarray, 
                        host_shear : np.ndarray, 
                        inclusion_bulk : np.ndarray, 
                        inclusion_shear : np.ndarray, 
                        aspect_ratio : np.ndarray) -> Tuple:
        """
        Calculate Eshelby's strain concentration factors P and Q.
        
        .. math::
            P = T/3 = \\frac{Tiijj}{3}
        .. math::
            Q = F/5 = \\frac{Tijij - Tiijj/3}{5}
        
        Parameters
        ----------
        host_bulk : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_bulk : np.ndarray
            Bulk modulus of inclusion component 
        inclusion_shear : np.ndarray
            Shear modulus of inclusion component 
        aspect_ratio : np.ndarray
            Aspect ratios  or scalar
            
        Returns
        -------
        P : np.ndarray
            Strain concentration factor P 
        Q : np.ndarray
            Strain concentration factor Q 
            
        Raises
        ------
        ValueError
            If input arrays have incompatible shapes
            If aspect ratio is exactly 1 (singularity in calculations)
            
            
        Notes
        -----
        - This function works for a single aspect ratio to avoid confusion
        - Implementation follows Keys and Xu (2002) Appendix A
        - Avoid aspect ratios exactly equal to 1 due to mathematical singularities
        
        References
        ----------
        .. [1] Keys, R.G. and Xu, S., 2002. An approximation for the Xu-White velocity model. Geophysics, 67(5), pp.1406-1414.
        .. [2] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """        
        A = (inclusion_shear/host_shear) - 1
        B = 1/3*(inclusion_bulk/host_bulk - inclusion_shear/host_shear)
        R = 3*host_shear / (3*host_bulk + 4*host_shear)

        aspect_ratio = rparray(aspect_ratio)
        aspect_ratio[aspect_ratio==1] = 1 - 0.00000001 #np.finfo(float).eps
        mask_below_one = aspect_ratio<1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nu_below_one = lambda aspect_ratio: aspect_ratio/((1-aspect_ratio**2)**(3/2)) * (np.arccos(aspect_ratio) - aspect_ratio*np.sqrt(1-aspect_ratio**2))
            nu_above_one = lambda aspect_ratio: aspect_ratio/((aspect_ratio**2-1)**(3/2)) * (aspect_ratio*np.sqrt(aspect_ratio**2-1) - np.arccosh(aspect_ratio))
            nu = np.where(mask_below_one, nu_below_one(aspect_ratio), nu_above_one(aspect_ratio))

        g = (3*nu-2) * (aspect_ratio**2)/(1-aspect_ratio**2)

        F1 = 1 + A*(3/2*(g+nu) - R*(3/2*g + 5/2*nu - 4/3))
        F2 = 1 + A*(1 + 3/2*(g+nu) - R/2*(3*g+5*nu)) + B*(3-4*R) + \
            A/2*(A+3*B)*(3-4*R)*(g+nu-R*(g-nu+2*nu**2))
        F3 = 1 + A/2*(R*(2-nu) + (1+aspect_ratio**2)/(aspect_ratio**2)*g*(R-1))
        F4 = 1 + A/4*(3*nu + g - R*(g-nu))
        F5 = A*(R*(g+nu-4/3) - g) + B*nu*(3-4*R)
        F6 = 1 + A*(1+g-R*(g+nu)) + B*(1-nu)*(3-4*R)
        F7 = 2 + A/4*(9*nu + 3*g - R*(5*nu+3*g)) + B*nu*(3-4*R)
        F8 = A*(1 - 2*R + g/2*(R-1) + nu/2*(5*R-3)) + B*(1-nu)*(3-4*R)
        F9 = A*(g*(R-1) - R*nu) + B*nu*(3-4*R)

        Tiijj = 3*F1/F2
        F = 2/F3 + 1/F4 + (F4*F5 + F6*F7 - F8*F9)/(F2*F4)
        
        return Tiijj, F

    @staticmethod
    def berryman_factors(host_bulk : np.ndarray, 
                         host_shear : np.ndarray, 
                         inclusion_bulk : np.ndarray, 
                         inclusion_shear : np.ndarray, 
                         pore_shape: ShapeName | Literal['SPHEROID','SPHERE', 'NEEDLE', 'DISK', 'CRACK'],
                         aspect_ratio : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Berryman's (1995) approximation to Eshelby's factors P and Q.
        
        Parameters
        ----------
        host_bulk : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_bulk : np.ndarray
            Bulk modulus of inclusion component 
        inclusion_shear : np.ndarray
            Shear modulus of inclusion component 
        pore_shape : ShapeName or str
            Inclusion shape specification. Valid values:
            - SPHEROID
            - SPHERE
            - NEEDLE
            - DISK
            - CRACK
        aspect_ratio : np.ndarray
            Aspect ratios  or (Nsamples, Ncomponents)
            
        Returns
        -------
        P : np.ndarray
            Strain concentration factor P 
        Q : np.ndarray
            Strain concentration factor Q 
            
        Notes
        -----
        - If shape is either Spheroidal (SPHEROID) or Penny crack ('CRACK), aspect ratio must be provided
        - For shape=0, results are identical to Eshelby's factors
        - Uses Berryman's 1995 mixture theories for rock properties
        - Small epsilon (eps) added to denominators to avoid division by zero
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Berryman, J.G., 1995. Mixture theories for rock properties. Rock physics and phase relations: A handbook of physical constants, 3, pp.205-228.

        """
        beta = lambda bulk, shear: shear   * (3*bulk +   shear) / (3*bulk + 4*shear)
        gamma =lambda bulk, shear: shear   * (3*bulk +   shear) / (3*bulk + 7*shear)
        zeta = lambda bulk, shear: shear/6 * (9*bulk + 8*shear) / (  bulk + 2*shear)
        
        match pore_shape.upper():
            case ShapeName.SPHERE:
                P = (host_bulk + 4/3*host_shear) /\
                    (inclusion_bulk + 4/3*host_shear)
                Q = (host_shear + zeta(host_bulk, host_shear)) /\
                    (inclusion_shear + zeta(host_bulk, host_shear))

            case ShapeName.NEEDLE:
                P = (host_bulk + host_shear + inclusion_shear/3)/\
                    (inclusion_bulk + host_shear + inclusion_shear/3)
                Q = 1/5* (   (4*host_shear)/\
                            (host_shear+inclusion_shear) +\
                        2*(host_shear+gamma(host_bulk,host_shear))/\
                            (inclusion_shear+gamma(host_bulk,host_shear)) +\
                            (inclusion_bulk+4/3*host_shear)/\
                            (inclusion_bulk+host_shear+inclusion_shear/3))

            case ShapeName.DISK:
                if 0 in inclusion_shear:
                    raise ValueError('DISKs can only contain non-zero shear inclusions.')
                
                P = (host_bulk + 4/3*inclusion_shear) / \
                    (inclusion_bulk + 4/3*inclusion_shear)
                Q = (host_shear + zeta(inclusion_bulk, inclusion_shear)) /\
                    (inclusion_shear + zeta(inclusion_bulk, inclusion_shear))

            case ShapeName.CRACK:
                """based on  Walsh (1969): assuming Ki<<Km and Gi<<Gm"""
                P = (host_bulk+4/3*inclusion_shear)/\
                    (inclusion_bulk+4/3*inclusion_shear+np.pi*aspect_ratio*beta(host_bulk, host_shear))
                Q = 1/5*(  1+\
                        (8*host_shear)/\
                        (4*inclusion_shear+np.pi*aspect_ratio*(host_shear+2*beta(host_bulk,host_shear)))+\
                        2*(inclusion_bulk+2/3*(inclusion_shear+host_shear))/\
                        (inclusion_bulk+4/3*inclusion_shear+np.pi*aspect_ratio*beta(host_bulk,host_shear)))
            case ShapeName.SPHEROID:
                Tiijj, F = InclusionMethods.eshelby_factors( host_bulk, host_shear, inclusion_bulk, inclusion_shear, aspect_ratio)
                P = Tiijj/3
                Q = F/5
            case _:
                raise ValueError("'{}' is not a valid shape. Shape must either be 'SPHERE', 'NEEDLE', 'DISK' or 'CRACK'.".format(pore_shape))
            
        return P, Q

    @staticmethod
    def multi_berryman_coefficients(host_bulk : np.ndarray, 
                                    host_shear : np.ndarray, 
                                    inclusion_list: List[Inclusion]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Berryman's (1995) factors P and Q for multiple inclusions.
        
        Parameters
        ----------
        host_bulk : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_list : List[inclusions]
            A list of Inclusions
            
        Returns
        -------
        P : np.ndarray
            Strain concentration factor P 
        Q : np.ndarray
            Strain concentration factor Q

        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Berryman, J.G., 1995. Mixture theories for rock properties. Rock physics and phase relations: A handbook of physical constants, 3, pp.205-228.
        
        """     
        P = np.array([])
        Q = np.array([])
        for inclusion in inclusion_list:
            p, q = InclusionMethods.berryman_factors(host_bulk, host_shear, inclusion.bulk, inclusion.shear, inclusion.shape_name, inclusion.aspect_ratio)
            P = np.vstack([P, p]) if P.size else p
            Q = np.vstack([Q, q]) if Q.size else q
        return P, Q

    @staticmethod
    def mori_tanaka(host_bulk : np.ndarray, 
                    host_shear : np.ndarray, 
                    inclusion_set: Dict[Inclusion, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mori-Tanaka's inclusion model for effective elastic moduli.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of inclusions (as keys) and corresponding porosity fractions (as value)
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 
            
        Raises
        ------
        ValueError
            If input arrays have incompatible shapes
            
            
        Notes
        -----
        - If shape is SPHEROID (Spheroidal) or CRACK (Penny crack), aspect ratio must be provided
        - For shape=0, results are identical to Kuster-Toksoz model
        - The Mori-Tanaka approach works best for:
            - Spheres and aligned ellipsoids when number of phases more than or equal 2
            - Randomly oriented ellipsoids when there are exactly two phases
        - Other configurations can be problematic (Torquato, 2002)
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Torquato, S., 2002. Random heterogeneous materials: microstructure and macroscopic properties. Springer Science & Business Media.

        """        
        nomK = denomK = nomG = denomG = inclusion_fraction = 0
        for inclusion, fraction in inclusion_set.items():
            p, q = InclusionMethods.berryman_factors(host_bulk, host_shear, inclusion.bulk, inclusion.shear, inclusion.shape_name, inclusion.aspect_ratio)
            nomK   += fraction * p * (inclusion.bulk - host_bulk)
            denomK += fraction * p
            nomG   += fraction * q * (inclusion.shear - host_shear)
            denomG += fraction * q
            inclusion_fraction += fraction

        host_fraction = 1-inclusion_fraction
        bulk = host_bulk + nomK/(host_fraction+denomK)
        shear = host_shear + nomG/(host_fraction+denomG)

        return bulk, shear

    @staticmethod
    def kuster_toksoz(host_bulk : np.ndarray, 
                      host_shear : np.ndarray, 
                      inclusion_set: Dict[Inclusion, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kuster-Toksoz's inclusion model for effective elastic moduli.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of inclusions (as keys) and corresponding porosity fractions (as value)
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 
            
        Raises
        ------
        ValueError
        If input arrays have incompatible shapes
        If aspect ratios contain value 1 (spherical inclusions not allowed)
        If Phi/aspect condition is violated
            
            
        Notes
        -----
        - Simulates fractures using spheroidal inclusions
        - Neglects multiple scattering effects
        - Assumes isolated inclusions where Phi/aspect << 1
        - Aspect ratios should not contain 1 (use spherical Berryman model instead)
        
        References
        ----------
        .. [1] Xu, S. and White, R.E., 1995. A new velocity model for clay-sand mixtures. Int. Journal of Rock Mechanics and Mining Sciences and Geomechanics Abstracts (Vol. 7, No. 32, p. 333A).
        .. [2] Keys, R.G. and Xu, S., 2002. An approximation for the Xu-White velocity model. Geophysics, 67(5), pp.1406-1414.
        .. [3] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """   
        SigmaK = SigmaG = 0
        zetam = (1/6)*host_shear * (9*host_bulk + 8*host_shear) / (host_bulk + 2*host_shear)
        for inclusion, fraction in inclusion_set.items():
            if not np.all(fraction/inclusion.aspect_ratio < 1.):
                warnings.warn('Results may be invalid. KUSTERTOKSOZ assumes isolated iclusions where phi/aspect<<1.')

            p, q = InclusionMethods.berryman_factors(host_bulk, host_shear, inclusion.bulk, inclusion.shear, inclusion.shape_name, inclusion.aspect_ratio)
            SigmaK += (inclusion.bulk - host_bulk) * fraction * p
            SigmaG += (inclusion.shear - host_shear) * fraction * q

        bulk = ((host_bulk + 4/3*host_shear)*host_bulk + 4/3*host_shear*SigmaK) / ((host_bulk + 4/3*host_shear) - SigmaK)
        shear = ((host_shear + zetam)*host_shear + zetam*SigmaG) / ((host_shear + zetam) - SigmaG)        
        return bulk, shear

    @staticmethod
    def self_consistent(host_bulk : np.ndarray, 
                        host_shear : np.ndarray, 
                        inclusion_set: Dict[Inclusion, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Self-Consistent effective medium based on Berryman's general form for multiple phases.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of inclusions (as keys) and corresponding porosity fractions (as value)
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 
            
        Raises
        ------
        ValueError
        If input arrays have incompatible shapes
        If volume fractions do not sum to approximately 1
            
            
        Notes
        -----
        - Consider to rearrange inputs such that there is no host and only inclusions are included, 
          because in self-cosistent model there is no preferred (host) material
        - If shape is SPHEROID (Spheroidal) or CRACK (Penny crack), aspect ratio must be provided
        - For shape=SPHEROID, results are identical to Kuster-Toksoz model
        - Uses iterative self-consistent scheme to solve for effective moduli
        - Initial guess uses Voigt average
        - Maximum of 10 iterations for convergence
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """        

        if len(inclusion_set)>0:        
            host_inclusion = Inclusion( ShapeName.SPHERE, 1., ElasticPropertySet(0, host_bulk, host_shear))
            host_fraction = np.full_like(rparray(host_bulk), 1.- np.array([*inclusion_set.values()]).sum(0))

            sc_inclusion_set = copy.deepcopy(inclusion_set)
            sc_inclusion_set[host_inclusion] = host_fraction

            bulks = rparray(np.broadcast_arrays(*[ inclusion.bulk for inclusion in sc_inclusion_set.keys()]))
            shears= rparray(np.broadcast_arrays(*[inclusion.shear for inclusion in sc_inclusion_set.keys()]))
            fractions = rparray(np.broadcast_arrays(*list(sc_inclusion_set.values())))

            bulk_eff = BoundMethods.voigt( bulks, fractions)
            shear_eff= BoundMethods.voigt(shears, fractions)

            inclusion_factors = lambda Keff, Geff, : InclusionMethods.multi_berryman_coefficients(Keff, Geff, list(sc_inclusion_set.keys()))
            for iter in range(10):
                P, Q = inclusion_factors(bulk_eff, shear_eff)
                bulk_eff = np.sum(bulks *fractions*P, axis=0)/np.sum(fractions*P, axis=0)
                shear_eff= np.sum(shears*fractions*Q, axis=0)/np.sum(fractions*Q, axis=0)
            bulk = bulk_eff 
            shear= shear_eff 
        
        else:
            bulk = host_bulk
            shear = host_shear
    
        return bulk, shear

    @staticmethod
    def dem(host_bulk : np.ndarray, 
            host_shear : np.ndarray, 
            inclusion_set: Dict[Inclusion, np.ndarray], 
            increments:int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Differential Effective Medium based on Kuster-Toksoz's model.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of inclusions (as keys) and corresponding porosity fractions (as value)
        increments : int
            Number of increments that the porosity is partitioned into.
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 
            
        Raises
        ------
        ValueError
        If input arrays have incompatible shapes
        If Phi values are invalid (negative or too large)
            
            
        Notes
        -----
        - Simulates fractures using spheroidal inclusions
        - Uses incremental inclusion process with 100 steps
        - At each step, a small fraction of inclusions is added to the current effective medium
        - The process continues until the total porosity Phi is reached

        References
        ----------
        .. [1] Berryman, J.G., 1980. Long-wavelength propagation in composite elastic media II. Ellipsoidal inclusions. The Journal of the Acoustical Society of America, 68(6), pp.1820-1831.
        .. [2] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """          
        bulk = host_bulk
        shear = host_shear
        for inclusion, fraction in inclusion_set.items():
            cumulative_fraction = 0
            fraction_increment = fraction/increments
            current_inclusion = copy.deepcopy(inclusion)
            for iter in range(increments):
                current_fraction = fraction_increment/(1-cumulative_fraction)
                bulk, shear = InclusionMethods.kuster_toksoz(bulk, shear, {current_inclusion:current_fraction})
                cumulative_fraction = cumulative_fraction + fraction_increment
        
        return bulk, shear                     

    @staticmethod
    def dem_modified(host_bulk : np.ndarray, 
                     host_shear : np.ndarray, 
                     inclusion : Dict[Inclusion, np.ndarray], 
                     critical_porosity : float = 0.4, 
                     increments : int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified Differential Effective Medium for rock with critical porosity.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of inclusions (as keys) and corresponding porosity fractions (as value)
        critical_porosity : float
            Critical Porosity of rock.
        increments : int
            Number of increments that the porosity is partitioned into.
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 
            
        Raises
        ------
        ValueError
        If input arrays have incompatible shapes
        If Phi values are invalid (negative or too large)
            
            
        Notes
        -----
        - Simulates fractures using spheroidal inclusions
        - Uses incremental inclusion process with 100 steps
        - At each step, a small fraction of inclusions is added to the current effective medium
        - The process continues until the total porosity Phi is reached

        References
        ----------
        .. [1] Berryman, J.G., 1980. Long-wavelength propagation in composite elastic media II. Ellipsoidal inclusions. The Journal of the Acoustical Society of America, 68(6), pp.1820-1831.
        .. [2] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """          

        if len(inclusion)>1:
            raise(ValueError('No more than a single inclusion is allowed for modified DEM'))
        
        critical_inclusion = copy.deepcopy(inclusion)
        critical_inclusion[0].bulk = BoundMethods.voigt_reuss_hill_average([host_bulk, inclusion[0].bulk], [1-critical_porosity, critical_porosity], upper_weight=0)
        critical_inclusion[0].shear = BoundMethods.voigt_reuss_hill_average([host_shear, inclusion[0].shear], [1-critical_porosity, critical_porosity], upper_weight=0)

        bulk, shear = InclusionMethods.dem( host_bulk, host_shear, critical_inclusion, increments)
        
        return bulk, shear  
    
    #Crack Methods==============================================
    class CrackMethodName(MultiEnum):
        DryPennyCracks = 'dry_penny_cracks'
        Hudson = 'hudson'
        EshelbyCheng = 'eshelby_cheng'

    @staticmethod
    def dry_penny_cracks(host_bulk, host_shear, crack_set: Dict[Inclusion, np.ndarray]):
        """
        Self-Consistent model for a cracked medium with randomly oriented dry penny-shaped cracks.
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of crack inclusions (as keys) and corresponding porosity fractions (as value)
     
        Returns
        -------
        bulk : np.ndarray
            Effective bulk modulus 
        shear : np.ndarray
            Effective shear modulus 

        Warnings
        --------
        RuntimeWarning
            If exact solution fails and approximate form is used

        Notes
        -----
        - Simulates fractures using spheroidal inclusions
        - Uses incremental inclusion process with 100 steps
        - At each step, a small fraction of inclusions is added to the current effective medium
        - The process continues until the total porosity Phi is reached

        References
        ----------
        .. [1] Berryman, J.G., 1980. Long-wavelength propagation in composite elastic media II. Ellipsoidal inclusions. The Journal of the Acoustical Society of America, 68(6), pp.1820-1831.
        .. [2] O'Connell, R.J. and Budiansky, B., 1974. Seismic velocities in dry and saturated cracked solids. Journal of geophysical Research, 79(35), pp.5412-5426.

        """          
        # host_bulk = np.atleast_1d(host_bulk).reshape(-1, 1) #convert into column vector
        # host_shear = np.atleast_1d(host_shear).reshape(-1, 1) #convert into column vector
        
        crack =  next(iter(crack_set.keys()))
        fraction = crack_set[crack]
        crack_density = crack_density(fraction, crack.aspect_ratio)
        host_poisson = conversions.modulus_to_poisson(host_bulk, host_shear)

        try:
            pnu = np.vstack([ 16*crack_density*(2*host_poisson+1), 
                             -(160*crack_density*host_poisson+45), 
                              (45*(host_poisson+2)-16*crack_density*(3*host_poisson+1)), 
                              (160*crack_density*host_poisson-90*host_poisson)]).T
            nu_raw = np.array([np.roots(c) for c in pnu])
            first_idx = np.argmax(nu_raw.real < 0.5, axis=1)
            nudry = nu_raw[np.arange(nu_raw.shape[0]), first_idx]

        except:
            nudry = host_poisson*(1-16/9*crack_density)
            warnings.warn('Due to an error in the exact soluion of Poisson ratio, the approximate form is used')

        bulk = host_bulk * (1-16/9*crack_density*(1-nudry**2)/(1-2*nudry))
        shear = host_shear* (1-32/45*crack_density*(1-nudry)*(5-nudry)/(2-nudry))

        return bulk, shear

    @staticmethod
    def hudson(host_bulk, host_shear, crack_set: Dict[Inclusion, np.ndarray], order: int = 1):
        """
        Hudson's model for Single crack set with normal along the 3rd-axis (Transversely isotropic media)
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of crack inclusions (as keys) and corresponding porosity fractions (as value)
        order: int
            Order of Hudson correction

     
        Returns
        -------
        (c11, c33, c44, c66, c12, c13) : tuple
            Effective Voigt matrix elements of dry rock 

        Notes
        -----
        - We can model following cases in Hudson model:
            - Dry rock: Where inclusion content is Dry.
            - Weak Crack:  Where inclusions contents are fluid-filled and aspect>0
            - 'closed':  Where inclusions contents are fluid-filled they are Infinitely thin (aspect~0) 

        References
        ----------
        .. [1] Berryman, J.G., 1980. Long-wavelength propagation in composite elastic media II. Ellipsoidal inclusions. The Journal of the Acoustical Society of America, 68(6), pp.1820-1831.
        .. [2] Cheng, C.H., 1993. Crack models for a transversely anisotropic medium. J. Geophys. Res., 98, 675–684.

        """ 
        crack =  next(iter(crack_set.keys()))
        fraction = crack_set[crack]
        crack_density = crack_density(fraction, crack.aspect_ratio)
        lame = conversions.modulus_to_Lame(host_bulk, host_shear)
        c11_0, c33_0, c44_0, c66_0, _, c13_0 = conversions.modulus_to_stiffness( host_bulk, host_shear )

        eps = np.finfo(float).eps
        kapa = (3*crack.bulk+4*crack.shear)/(3*np.pi*crack.aspect_ratio*host_shear+eps) * (lame+2*host_shear)/(lame+host_shear)
        M = (4*crack.shear/(np.pi*crack.aspect_ratio*host_shear+eps)) * (lame+2*host_shear)/(3*lame+4*host_shear)
        U3 = (4/3)*(lame+2*host_shear)/(lame+host_shear) /(1+kapa)
        U1 = (16/3)*(lame+2*host_shear)/(3*lame+4*host_shear) /(1+M)

        q = 15*(lame/host_shear)**2 + 28*(lame/host_shear) + 28

        c11_1 = -(lame**2)/host_shear * crack_density * U3
        c13_1 = -(lame*(lame+2*host_shear))/host_shear * crack_density * U3
        c33_1 = -((lame+2*host_shear)**2)/host_shear * crack_density * U3
        c44_1 = -host_shear*crack_density*U1
        c66_1 = 0

        c11_2 = (q/15) * (lame**2)/(lame+2*host_shear) * (crack_density * U3)**2
        c13_2 = (q/15) * lame * (crack_density * U3)**2
        c33_2 = (q/15) * (lame+2*host_shear) * (crack_density * U3)**2
        c44_2 = (2/15) * host_shear*(3*lame+8*host_shear)/(lame+2*host_shear) * (crack_density * U1)**2
        c66_2 = 0

        match order:
            case 1:
                c11 = c11_0 + c11_1
                c33 = c33_0 + c33_1
                c13 = c13_0 + c13_1
                c44 = c44_0 + c44_1
                c66 = c66_0 + c66_1
            case 2:
                c11 = c11_0 + c11_1 + c11_2
                c33 = c33_0 + c33_1 + c33_2
                c44 = c44_0 + c44_1 + c44_2
                c66 = c66_0 + c66_1 + c66_2
                c13 = c13_0 + c13_1 + c13_2
        c12 = c11 - 2*c44

        return c11, c33, c44, c66, c12, c13

    @staticmethod
    def eshelby_cheng (host_bulk, host_shear, crack_set: Dict[Inclusion, np.ndarray]):
        """
        Eshelby-Cheng's model for fluid-filled ellipsoidal crack set with normal along the 3rd-axis (Transversely isotropic media)
        
        Parameters
        ----------
        host_bulk     : np.ndarray
            Bulk modulus of matrix (host component) 
        host_shear    : np.ndarray
            Shear modulus of matrix (host component) 
        inclusion_set : np.ndarray
            A dictionary of crack inclusions (as keys) and corresponding porosity fractions (as value)
     
        Returns
        -------
        (c11, c33, c44, c66, c12, c13) : tuple
            Effective Voigt matrix elements of dry rock 

        Notes
        -----
        - The model is appropriate for high-frequency laboratory conditions.
        - For low-frequency field situations, use Hudson's dry equations and then saturate by using the Brown 
          and Korringa relations

        References
        ----------
        .. [1] Berryman, J.G., 1980. Long-wavelength propagation in composite elastic media II. Ellipsoidal inclusions. The Journal of the Acoustical Society of America, 68(6), pp.1820-1831.
        .. [2] Cheng, C.H., 1993. Crack models for a transversely anisotropic medium. J. Geophys. Res., 98, 675–684.

        """ 

        crack =  next(iter(crack_set.keys()))
        fraction = crack_set[crack]
        L = conversions.modulus_to_Lame(host_bulk, host_shear)
        c11_0, c33_0, c44_0, c66_0, _, c13_0 = conversions.modulus_to_stiffness( host_bulk, host_shear )

        sig = (3*host_bulk-2 * host_shear) / (6*host_bulk + 2*host_shear)
        Sa = np.sqrt(1 - crack.aspect_ratio**2)
        R = (1 - 2*sig) / (8*np.pi*(1-sig))
        Q = 3*R / (1 - 2*sig)
        Ia = 2*np.pi * crack.aspect_ratio * (np.arccos(crack.aspect_ratio) - crack.aspect_ratio * Sa) / (Sa**3)
        Ic = 4*np.pi - 2*Ia
        Iac = (Ic - Ia) / (3*Sa**2)
        Iaa = np.pi - 3*Iac/4
        Iab = Iaa/3

        S11 = Q*Iaa + R*Ia
        S33 = Q*(4*np.pi/3 - 2*Iac*crack.aspect_ratio**2) + Ic*R
        S12 = Q*Iab - R*Ia
        S13 = Q*Iac*crack.aspect_ratio**2 - R*Ia
        S31 = Q*Iac - R*Ic
        S1212 = Q*Iab + R*Ia
        S1313 = Q*(1+crack.aspect_ratio**2)*Iac/2 + R*(Ia+Ic)/2

        C = crack.bulk/(3*(host_bulk-crack.bulk))
        D = S33*S11 + S33*S12 - 2*S31*S13 - (S11+S12+S33-1-3*C) - C*(S11+S12+2*(S33-S13-S31))
        E = S33*S11 - S31*S13 - (S33+S11-2*C-1) + C*(S31+S13-S11-S33)

        c11_1 = L*(S31-S33+1) + 2*host_shear*E/(D*(S12-S11+1))
        c33_1 = ((L+2*host_shear)*(-S12-S11+1) + 2*L*S13 + 4*host_shear*C)/D
        c13_1 = ((L+2*host_shear)*(S13+S31) - 4*host_shear*C + L*(S13-S12-S11-S33+2))/(2*D)
        c44_1 = host_shear/(1-2*S1313)
        c66_1 = host_shear/(1-2*S1212)

        c11 = c11_0 - fraction*c11_1
        c33 = c33_0 - fraction*c33_1
        c44 = c44_0 - fraction*c44_1
        c66 = c66_0 - fraction*c66_1
        c13 = c13_0 - fraction*c13_1
        c12 = c11 - 2*c44

        return c11, c33, c44, c66, c12, c13
