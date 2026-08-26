"""
Bounds and averaging  relations for multiphase mixtures.

This module provides rock physics methods for calculating theoretical bounds and
averages of elastic properties (structure_unaware models) in multiphase composite 
materials. The bounds represent theoretical limits between which actual effective 
properties must lie, while the averaging methods provide practical estimates based on 
mixing rules.
"""

import warnings
import numpy as np
from rokpy.utilities import MultiEnum, rparray


class BoundMethods():
    class MixingMethodName(MultiEnum):
        """
        Names of available Bound Methods for mixing several components, such as minerals or fluids
        """    
        VoigtReussHill = 'voigt_reuss_hill', 'vrh'
        HashinShtrikmanWalpole = 'hashin_shtrikman_walpole', 'hs'

    class RockMethodName(MultiEnum):
        """
        Names of available Bound Methods for mixing rock's solid with pore/fluid phase
        """  
        VoigtReussHill = 'modified_voigt_reuss_hill', 'vrh'
        HashinShtrikman = 'modified_hashin_shtrikman', 'hs'
    #----------------------------------
    @staticmethod
    def bounds_average(upper_bound: np.ndarray, 
                       lower_bound: np.ndarray, 
                       upper_weight: np.ndarray) -> np.ndarray:
        """
        Weighted average of uppler and lower bounds

        Parameters
        ----------
        upper_bound : np.ndarray
            Upper bound, e.g. Voigt bound or Hashin-Shtrikman upper bound
        lower_bound : np.ndarray
            Upper bound, e.g. Reuss bound or Hashin-Shtrikman lower bound
        upper_weight : np.ndarray
            Averaging weight of upper bound

        Returns
        -------
        average : np.ndarray
            Average of upper and lower bounds
        """
        return upper_weight*upper_bound + (1-upper_weight)*lower_bound

    @staticmethod
    def voigt(properties: np.ndarray, 
              fractions: np.ndarray) -> np.ndarray:
        """
        Voigt's upper bound for given properties sets

        Parameters
        ----------
        properties : np.ndarray
            An stacked array of properties
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties

        Returns
        -------
        average: np.ndarray
            Voigt average of properties

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        result = 0
        for property, fraction in zip(properties, fractions):
            result += property*fraction
        return result

    @staticmethod
    def reuss(properties: np.ndarray, 
              fractions: np.ndarray) -> np.ndarray:
        """
        Reuss's upper bound for given properties sets

        Parameters
        ----------
        properties : np.ndarray
            An stacked array of properties
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties

        Returns
        -------
        average: np.ndarray
            Reuss average of properties

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        result = 0
        for property, fraction in zip(properties,fractions):
            result += fraction/(property + np.finfo(float).eps)
        return 1/result
        # return np.array(1 / np.sum( np.array(properties, ndmin=1) / (np.array(fractions, ndmin=1)+np.finfo(float).eps) ), ndmin=1 )

    @staticmethod
    def voigt_reuss_hill_average(properties: np.ndarray, 
                                 fractions: np.ndarray, 
                                 upper_weight: np.ndarray = 0.5):
        """
        Hill's average of Voigt (upper) and Reuss (Lower) bounds for given set of properties

        Parameters
        ----------
        properties : np.ndarray
            An stacked array of properties
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties
        upper_weight : np.ndarray
            Averaging weight of upper (Voigt) bound, default is 0.5
        
        Returns
        -------
        average: np.ndarray
            Voigt-Reuss-Hill average of properties

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        """
        return BoundMethods.bounds_average(BoundMethods.voigt(properties, fractions), BoundMethods.reuss(properties, fractions), upper_weight)

    @staticmethod
    def hashin_shtrikman_walpole_lower(bulks: np.ndarray, 
                                       shears: np.ndarray,  
                                       fractions: np.ndarray) -> np.ndarray:
        """
        Hashin-Shtrikman-Walpole Lower bounds for given set of non-well-ordered bulk and shear moduli

        Parameters
        ----------
        bulks : np.ndarray
            An stacked array of bulk moduli
        shears : np.ndarray
            An stacked array of shear moduli
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties
        
        Returns
        -------
        bulk_lower: np.ndarray
            Lower bound of Hashin-Shtrikman bulk modulus
        shear_lower: np.ndarray
            Lower bound of Hashin-Shtrikman shear modulus

        NOTES
        -----
        - If materials with higher bulk modulus also have higher shear modulus, they are said to be Well-orderd.
          For example Quartz and Calcite are not well-ordered because Kqz < Kcal, while Gqz > Gcal.
        - This is the main difference between Hashin-Shtrikman (which assumes the components to be well ordered)  
          and Hashin-Shtrikman-Walpole (which does not need the components to be well-ordered)

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        bulks = rparray(bulks)
        shears = rparray(shears)
        fractions = rparray(fractions)

        bulk_min = bulks.min(axis=0) #Find the minimum value among the elements (at each sample)
        shear_min = shears.min(axis=0)

        zeta = lambda K, G: G/6*(9*K + 8*G) / (K + 2*G)
        Gamma = lambda z: BoundMethods.reuss(shears + z, fractions) - z
        Lambda = lambda z: BoundMethods.reuss(bulks + 4/3*z, fractions) - 4/3*z

        bulk_lower = Lambda(shear_min)
        shear_lower = Gamma(zeta(bulk_min, shear_min))

        return bulk_lower, shear_lower

    @staticmethod
    def hashin_shtrikman_walpole_upper(bulks: np.ndarray, 
                                       shears: np.ndarray,  
                                       fractions: np.ndarray):
        """
        Hashin-Shtrikman-Walpole upper bounds for given set of non-well-ordered bulk and shear moduli

        Parameters
        ----------
        bulks : np.ndarray
            An stacked array of bulk moduli
        shears : np.ndarray
            An stacked array of shear moduli
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties
        
        Returns
        -------
        bulk_upper: np.ndarray
            Upper bound of Hashin-Shtrikman bulk modulus
        shear_upper: np.ndarray
            Upper bound of Hashin-Shtrikman shear modulus

        NOTES
        -----
        - If materials with higher bulk modulus also have higher shear modulus, they are said to be Well-orderd.
          For example Quartz and Calcite are not well-ordered because Kqz < Kcal, while Gqz > Gcal.
        - This is the main difference between Hashin-Shtrikman (which assumes the components to be well ordered)  
          and Hashin-Shtrikman-Walpole (which does not need the components to be well-ordered)

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        """
        bulks = rparray(bulks)
        shears = rparray(shears)
        fractions = rparray(fractions)

        bulk_max = bulks.max(axis=0)
        shear_max = shears.max(axis=0)

        zeta = lambda K, G: G/6*(9*K + 8*G) / (K + 2*G)
        Gamma = lambda z: BoundMethods.reuss(shears + z, fractions) - z
        Lambda = lambda z: BoundMethods.reuss(bulks + 4/3*z, fractions) - 4/3*z

        bulk_upper = Lambda(shear_max)
        shear_upper = Gamma(zeta(bulk_max, shear_max))

        return bulk_upper, shear_upper

    @staticmethod
    def voigt_reuss_hill(bulks: np.ndarray, 
                         shears: np.ndarray,  
                         fractions: np.ndarray, 
                         upper_weight = 0.5):
        """
        Voigt-Reuss-Hill average of given set of bulk and shear moduli

        Parameters
        ----------
        bulks : np.ndarray
            An stacked array of bulk moduli
        shears : np.ndarray
            An stacked array of shear moduli
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties
        upper_weight : np.ndarray
            Averaging weight of upper bound, default is 0.5
        
        Returns
        -------
        bulk: np.ndarray
            Voigt-Reuss-Hill average of bulk modulus
        shear: np.ndarray
            Voigt-Reuss-Hill average of shear modulus
        """ 
        bulk_upper = BoundMethods.voigt(bulks, fractions)
        shear_upper= BoundMethods.voigt(shears, fractions)
        bulk_lower = BoundMethods.reuss(bulks, fractions)
        shear_lower= BoundMethods.reuss(shears, fractions)
        
        bulk = BoundMethods.bounds_average(bulk_upper, bulk_lower, upper_weight)
        shear = BoundMethods.bounds_average(shear_upper, shear_lower, upper_weight)
        return bulk, shear

    @staticmethod
    def hashin_shtrikman_walpole(bulks: np.ndarray, 
                                 shears: np.ndarray,  
                                 fractions: np.ndarray, 
                                 upper_weight: np.ndarray = 0.5):
        """
        Hashin-Shtrikman-Walpole average for given set of non-well-ordered bulk and shear moduli

        Parameters
        ----------
        bulks : np.ndarray of shape (n_elements, n_samples)
            An stacked array of bulk moduli
        shears : np.ndarray
            An stacked array of shear moduli
        fractions : np.ndarray
            An stacked array of fractions corresponding to properties
        upper_weight : float or np.ndarray
            Averaging weight of upper bound, default is 0.5
            This property is generally a float value but it can also receive an np.ndarray of float to study the results for set or series of values.
        
        Returns
        -------
        bulk: np.ndarray
            Hashin-Shtrikman average of bulk modulus
        shear: np.ndarray
            Hashin-Shtrikman average of shear modulus

        NOTES
        -----
        - If materials with higher bulk modulus also have higher shear modulus, they are said to be Well-orderd.
          For example Quartz and Calcite are not well-ordered because Kqz < Kcal, while Gqz > Gcal.
        - This is the main difference between Hashin-Shtrikman (which assumes the components to be well ordered)  
          and Hashin-Shtrikman-Walpole (which does not need the components to be well-ordered)

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        """        
        bulk_upper, shear_upper = BoundMethods.hashin_shtrikman_walpole_upper(bulks, shears, fractions)
        bulk_lower, shear_lower = BoundMethods.hashin_shtrikman_walpole_lower(bulks, shears, fractions)
        
        bulk = BoundMethods.bounds_average(bulk_upper, bulk_lower, upper_weight)
        shear = BoundMethods.bounds_average(shear_upper, shear_lower, upper_weight)

        return bulk, shear

    @staticmethod
    def hashin_shtrikman_lower(stiff_bulk: np.ndarray, 
                               stiff_shear: np.ndarray, 
                               loose_bulk: np.ndarray, 
                               loose_shear: np.ndarray, 
                               loose_fraction: np.ndarray ):
        """
        Lower bound of Hashin-Shtrikman model for a two-phase set of well-ordered bulk and shear moduli

        Parameters
        ----------
        stiff_bulks : np.ndarray
            bulk modulus of stiffer material
        stiff_shears : np.ndarray
            shear modulus of stiffer material
        loose_bulks : np.ndarray
            bulk modulus of looser material
        loose_shears : np.ndarray
            shear modulus of looser material
        loose_fraction : np.ndarray
            fractions of loose component.
        
        Returns
        -------
        bulk_lower: np.ndarray
            Lower bound of Hashin-Shtrikman bulk modulus
        shear_lower: np.ndarray
            Lower bound of Hashin-Shtrikman shear modulus

        NOTES
        -----
        - If materials with higher bulk modulus also have higher shear modulus, they are said to be Well-orderd.
          For example Quartz and Calcite are not well-ordered because Kqz < Kcal, while Gqz > Gcal.
        - This is the main difference between Hashin-Shtrikman (which assumes the components to be well ordered)  
          and Hashin-Shtrikman-Walpole (which does not need the components to be well-ordered)

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        if np.any((stiff_bulk - loose_bulk) * (stiff_shear - loose_shear)<0):
            warnings.warn('Phases are not well-ordered. Hashin-Shtrikman results may be invalid. Use "hashin_shtrikman_walpole" method, instead.')

        bulk = lambda k1, k2, g1, f2: k1 + f2/(1/(k2-k1) + (1-f2)/(k1+4*g1/3))
        shear = lambda g1, g2, k1, f2: g1 + f2/(1/(g2-g1) + 2*(1-f2)*(k1+2*g1)/(5*g1*(k1+4/3*g1)))

        bulk_lower = bulk(loose_bulk, stiff_bulk, loose_shear, 1-loose_fraction)
        shear_lower = shear(loose_shear, stiff_shear,loose_bulk, 1-loose_fraction)
        return bulk_lower, shear_lower

    @staticmethod
    def hashin_shtrikman_upper(stiff_bulk: np.ndarray, 
                               stiff_shear: np.ndarray, 
                               loose_bulk: np.ndarray, 
                               loose_shear: np.ndarray, 
                               loose_fraction: np.ndarray ):
        """
        Upper bound of Hashin-Shtrikman model for a two-phase set of well-ordered bulk and shear moduli

        Parameters
        ----------
        stiff_bulks : np.ndarray
            bulk modulus of stiffer material
        stiff_shears : np.ndarray
            shear modulus of stiffer material
        loose_bulks : np.ndarray
            bulk modulus of looser material
        loose_shears : np.ndarray
            shear modulus of looser material
        loose_fraction : np.ndarray
            fractions of loose component.
        
        Returns
        -------
        bulk_upper: np.ndarray
            Upper bound of Hashin-Shtrikman bulk modulus
        shear_upper: np.ndarray
            Upper bound of Hashin-Shtrikman shear modulus

        NOTES
        -----
        - If materials with higher bulk modulus also have higher shear modulus, they are said to be Well-orderd.
          For example Quartz and Calcite are not well-ordered because Kqz < Kcal, while Gqz > Gcal.
        - This is the main difference between Hashin-Shtrikman (which assumes the components to be well ordered)  
          and Hashin-Shtrikman-Walpole (which does not need the components to be well-ordered)

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        if np.any((stiff_bulk - loose_bulk) * (stiff_shear - loose_shear)<0):
            warnings.warn('Phases are not well-ordered. Hashin-Shtrikman results may be invalid. Use "hashin_shtrikman_walpole" method, instead.')

        bulk = lambda k1, k2, g1, f2: k1 + f2/(1/(k2-k1) + (1-f2)/(k1+4*g1/3))
        shear = lambda g1, g2, k1, f2: g1 + f2/(1/(g2-g1) + 2*(1-f2)*(k1+2*g1)/(5*g1*(k1+4/3*g1)))

        bulk_upper = bulk(stiff_bulk, loose_bulk, stiff_shear, loose_fraction)
        shear_upper = shear(stiff_shear, loose_shear,stiff_bulk, loose_fraction)
        return bulk_upper, shear_upper

    @staticmethod
    def modified_hashin_shtrikman(mineralset_bulk: np.ndarray, 
                                  mineralset_shear: np.ndarray, 
                                  fluidset_bulk: np.ndarray, 
                                  fluidset_shear: np.ndarray, 
                                  porosity : np.ndarray, 
                                  critical_porosity: float = 1.0, 
                                  upper_weight = 0.5):
        """
        Modified average Hashin-Shtrikman model for bulk and shear moduli of a set of rock solid and fluid components

        Parameters
        ----------
        mineralset_bulk : np.ndarray
            bulk modulus of rock solid phase (mineralset)
        mineralset_shear : np.ndarray
            shear modulus of rock solid phase (mineralset)
        fluidset_bulk : np.ndarray
            bulk modulus of rock non-solid phase (fluidset or pore content)
        fluidset_shear : np.ndarray
            shear modulus of rock non-solid phase (fluidset or pore content)
        porosity : np.ndarray
            Porosity
        critical_porosity : float
            critical porosity, default is 1.0
        upper_weight : np.ndarray
            Averaging weight of upper bound
        
        Returns
        -------
        bulk: np.ndarray
            Effective bulk modulus
        shear: np.ndarray
            Effective shear modulus

        NOTES
        -----
        - Modified hasin-Shtrikman model only applicable for two-phase well ordered set of properties, e.g. the
          solid and fluid phases of a rock.

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        phip = porosity/critical_porosity
        bulk_critical, shear_critical   = BoundMethods.hashin_shtrikman_lower( mineralset_bulk, mineralset_shear, fluidset_bulk, fluidset_shear, critical_porosity )
        bulk_lower, shear_lower         = BoundMethods.hashin_shtrikman_lower( mineralset_bulk, mineralset_shear, bulk_critical, shear_critical, phip )
        bulk_upper, shear_upper         = BoundMethods.hashin_shtrikman_upper( mineralset_bulk, mineralset_shear, bulk_critical, shear_critical, phip )
        Klh, Glh                        = BoundMethods.hashin_shtrikman_lower( mineralset_bulk, mineralset_shear, fluidset_bulk, fluidset_shear, porosity )

        bulk_upper, bulk_lower, shear_upper, shear_lower, Klh, Glh, porosity = \
                np.atleast_1d(bulk_upper, bulk_lower, shear_upper, shear_lower, Klh, Glh, porosity)

        bulk_upper[porosity>critical_porosity] = Klh[porosity>critical_porosity]
        bulk_lower[porosity>critical_porosity] = Klh[porosity>critical_porosity]
        shear_upper[porosity>critical_porosity] = Glh[porosity>critical_porosity]
        shear_lower[porosity>critical_porosity] = Glh[porosity>critical_porosity]

        bulk = BoundMethods.bounds_average(bulk_upper, bulk_lower, upper_weight)
        shear = BoundMethods.bounds_average(shear_upper, shear_lower, upper_weight)

        return bulk, shear

    @staticmethod
    def modified_voigt_reuss_hill(mineralset_bulk: np.ndarray, 
                                  mineralset_shear: np.ndarray, 
                                  fluidset_bulk: np.ndarray, 
                                  fluidset_shear: np.ndarray, 
                                  porosity : np.ndarray, 
                                  critical_porosity: float = 1, 
                                  upper_weight: np.ndarray = 0.5):
        """
        Modified average Hashin-Shtrikman model for bulk and shear moduli of a set of rock solid and fluid components

        Parameters
        ----------
        mineralset_bulk : np.ndarray
            bulk modulus of rock solid phase (mineralset)
        mineralset_shear : np.ndarray
            shear modulus of rock solid phase (mineralset)
        fluidset_bulk : np.ndarray
            bulk modulus of rock non-solid phase (fluidset or pore content)
        fluidset_shear : np.ndarray
            shear modulus of rock non-solid phase (fluidset or pore content)
        porosity : np.ndarray
            Porosity
        critical_porosity : float
            critical porosity, default is 1.0
        upper_weight : np.ndarray
            Averaging weight of upper bound
        
        Returns
        -------
        bulk: np.ndarray
            Effective bulk modulus
        shear: np.ndarray
            Effective shear modulus

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        porosity = rparray(porosity)
        critical_porosity = rparray(critical_porosity)
        phip = porosity/critical_porosity
        bulk_critical = BoundMethods.reuss([mineralset_bulk, fluidset_bulk], [1-critical_porosity, critical_porosity])
        shear_critical = BoundMethods.reuss([mineralset_shear, fluidset_shear], [1-critical_porosity, critical_porosity])

        bulk_lower = BoundMethods.reuss([mineralset_bulk, fluidset_bulk], [1-porosity, porosity])
        shear_lower= BoundMethods.reuss([mineralset_shear, fluidset_shear], [1-porosity, porosity])

        bulk_upper = BoundMethods.voigt([mineralset_bulk, bulk_critical],[1-phip, phip])
        shear_upper = BoundMethods.voigt([mineralset_shear, shear_critical],[1-phip, phip])

        bulk_upper[porosity>critical_porosity] = bulk_lower[porosity>critical_porosity]
        shear_upper[porosity>critical_porosity] = shear_lower[porosity>critical_porosity]

        bulk = BoundMethods.bounds_average(bulk_upper, bulk_lower, upper_weight)
        shear = BoundMethods.bounds_average(shear_upper, shear_lower, upper_weight)
        return bulk, shear
