"""**Contains models simulating the rock and its components**

This module provides comprehensive object-oriented representations of rock physics
materials, from basic elastic solids to complex reservoir rocks. Materials are
hierarchically organized with lazy evaluation of derived properties, supporting
mixing laws, fluid substitution, and multiple effective-medium theories.


Examples
--------

Define a random material with given elastic parameters:

>>> from rokpy.materials import Material

>>> material = Material(p_velocity=3500, s_velocity=2000, density=2.2) # Define a material with given elastic parameters
>>> 
>>> #Access different elastic properties in material object
>>> print(f'Bulk Modulus        = {material.bulk}')
>>> print(f'Poisson Ratio       = {material.poisson}')
>>> print(f'Young Modulus       = {material.young}')

define a component (Mineral or Fluid) with given property set:

>>> from rokpy.constants import ElasticPropertySet
>>> clay_properties = ElasticPropertySet(density=2.35, bulk=25.0, shear=12.0, type='clay')
>>> clay = Mineral(clay_properties)
>>> print(clay)

or create a component (Mineral or Fluid) from a minerals/fluids property table

>>> from rokpy.constants import MineralsPropertyTable, FluidsPropertyTable
>>> 
>>> minerals_table = MineralsPropertyTable() #Realizing a table of default materials as an object
>>> quartz = minerals_table.Quartz           #Access the Quartz property set from the table
>>> print(quartz)
>>> fluids_table = FluidsPropertyTable()     #Realizing a table of default fluids as an object
>>> brine = fluids_table.Brine               #Access the brine property set from the table
>>> print(brine)

Create MneralSet and FluidSet objects:

>>> from rokpy.materials import MineralSet, FluidSet
>>> mineral_set = MineralSet({'clay': 0.3, 'quartz': 0.7})
>>> fluid_set = FluidSet(fluid_set={'Brine': 1.0})
>>> print(mineral_set)
>>> print(fluid_set)

create a Rock object:

>>> from rokpy.materials import InclusionRock
>>> rock = InclusionRock(mineral_set=mineral_set, fluid_set=fluid_set, total_porosity=0.2, rock_frame_method='dem')
>>> print(rock)

All other rock properties are automatically accessible as rock attributes:

>>> print(f'Rock Bulk Modulus        = {rock.bulk}')
>>> print(f'Rock Poisson Ratio       = {rock.poisson}')
>>> print(f'Rock Young Modulus       = {rock.young}')
>>> print(f'Rock Lame Coef:          = {rock.lame}')

"""

from rokpy import conversions
from rokpy.avo import ricker
from rokpy.constants import FluidsPropertyTable, FluidType, ElasticPropertySet, Color
from rokpy.effective_medium import Inclusion, BoundMethods, ContactMethods, InclusionMethods, FluidEffectMethods
from rokpy.models import DensityModel, MixingModel, InclusionModel, ContactModel
from rokpy.utilities import biot_characteristic_frequency, rparray
import numpy as np
import random
from copy import deepcopy
from enum import Enum
from typing import List, Dict, Literal, Tuple

class MaterialType(Enum):
    """Enumeration of high-level material categories used by the package.

    Members
    -------
    General, Mineral, Fluid, Matrix, Component, ComponentSet, Rock
        Semantic categories for organizing material objects.
    """
    General = 'GENERAL'
    Mineral = 'MINERAL'
    Fluid = 'FLUID'
    Matrix = 'MATRIX'
    Component = 'COMPONENT'
    ComponentSet = 'COMPONENTSET'
    Rock = 'ROCK'

    def __str__(self):
        return self.name

#===============================================================================
class Material:
    """
    Base class representing a generic material with elastic and density properties.

    A Material stores primary properties (p_velocity, s_velocity, density) and
    lazily (dynamically) computes derived elastic quantities (bulk/shear moduli, Poisson's
    ratio, Young's modulus, Lame constants) using the conversions module.

    Parameters
    ----------
    p_velocity : array-like or scalar
        Compressional wave velocity (m/s).
    s_velocity : array-like or scalar
        Shear wave velocity (m/s).
    density : array-like or scalar
        Bulk density (g/cc).
    type : MaterialType or str, optional
        Semantic type of the material. Default: MaterialType.General.
    color : tuple(int,int,int), optional
        RGB color used for plotting/visualization.

    Notes
    -----
    - Input values are converted to numpy arrays using utilities.rparray.
    - Other elastic properties (bulk, shear, p_modulus, poisson, young, lame)
      are computed when accessed.
    - Computing the properties rather than storing them ensures consistency.
    """
    def __init__(   self, 
                    p_velocity, 
                    s_velocity, 
                    density, 
                    type: str = MaterialType.General) -> None:
        self.type = type
        self.color = Color()
        self.density = density
        self.p_velocity = p_velocity
        self.s_velocity = s_velocity
        self.id = random.getrandbits(128)
        self.porosity_weight = 1

    @property
    def bulk(self):
        """Bulk modulus in GPa computed from p_velocity, s_velocity and density."""
        return conversions.velocity_to_bulk(self.p_velocity, self.s_velocity, self.density)

    @property
    def shear(self):
        """Shear modulus in GPa computed from s_velocity and density."""
        return conversions.velocity_to_shear(self.s_velocity, self.density)

    @property
    def p_modulus(self):
        """P-wave modulus (GPa) equal to K + 4/3 G."""
        return self.bulk + 4/3*self.shear

    @property
    def p_velocity(self):
        """P-wave velocity in m/s (stored primary parameter)."""
        return self._p_velocity
    @p_velocity.setter
    def p_velocity(self, value):
        self._p_velocity = rparray(value)

    @property
    def s_velocity(self):
        """S-wave velocity in m/s (stored primary parameter)."""
        return self._s_velocity
    @s_velocity.setter
    def s_velocity(self, value):
        self._s_velocity = rparray(value)

    @property
    def density(self):
        """Bulk density in g/cc (stored primary parameter)."""
        return self._density
    @density.setter
    def density(self, value):
        self._density = rparray(value)

    @property
    def type(self):
        """Material type (MaterialType or string)."""
        return self._type
    @type.setter
    def type(self, value):
        self._type = value

    @property
    def color(self):
        """RGB color tuple used for visualization."""
        return self._color
    @color.setter
    def color(self, value):
        self._color = value

    @property
    def poisson(self):
        """Poisson's ratio (unitless) computed from bulk and shear moduli."""
        return conversions.modulus_to_poisson(self.bulk, self.shear)

    @property
    def velocity_ratio(self):
        """P-to-S velocity ratio (v_p / v_s), unitless."""
        return conversions.velocity_ratio(self.p_velocity, self.s_velocity)

    @property
    def lame(self):
        """Lame's first parameter (lambda) in GPa."""
        return conversions.modulus_to_Lame(self.bulk, self.shear)

    @property
    def young(self):
        """Young's modulus in GPa computed from bulk and shear moduli."""
        return conversions.modulus_to_young(self.bulk, self.shear)

    @property
    def porosity_weight(self):
        """Porosity weight (dimensionless) used when combining minerals for pore-space allocation."""
        return self._porosity_weight
    @porosity_weight.setter
    def porosity_weight(self, value):
        self._porosity_weight = value

    def properties(self):
        """Return a tuple of key properties of the material (p_velocity, s_velocity, density)."""
        return self.p_velocity, self.s_velocity, self.density

    def moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bulk, shear) moduli in GPa computed from stored velocities and density.

        Returns
        -------
        (bulk, shear) : tuple of array-like
            Bulk and shear moduli (GPa).
        """
        return self.bulk, self.shear

    def velocity_Density_set(self):
        return (self.p_velocity, self.s_velocity, self.density)

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        text = '{:20s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}\n'.format('Type', 'Density', 'P-velocity', 'S-velocity', 'Bulk', 'Shear', 'Poisson')
        text += '-'*105 + '\n'
        text += '{:20s}|{:^15.3f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.3f}\n'.format(self.type, 
                                                                                              self.density.mean(), 
                                                                                              self.p_velocity.mean(), 
                                                                                              self.s_velocity.mean(),
                                                                                              self.bulk.mean(),
                                                                                              self.shear.mean(),
                                                                                              self.poisson.mean())
        return text 
    
    def __eq__(self, other):
        return other.p_velocity == self.p_velocity and other.s_velocity == self.s_velocity and other.density == self.density

#===============================================================================
class Component(Material):
    """
    Single-component material that gets its properties from an ElasticPropertySet.

    Components represent atomic constituents (a mineral end-member or a fluid)
    and expose properties directly from the associated ElasticPropertySet.

    Parameters
    ----------
    property_set : ElasticPropertySet
        Container describing bulk, shear, density, color and type for the component.

    Notes
    -----
    - The Component does not allow direct mutation of bulk/shear/density; these
      are accessed from the underlying property_set.
    """
    def __init__( self, 
                  property_set: ElasticPropertySet) -> None:
        self.type = property_set.type
        self.properties = property_set
        self.id = random.getrandbits(128)

    @property
    def bulk(self):
        """Bulk modulus (GPa) taken from the component property set."""
        return rparray(self.properties.bulk)
    @bulk.setter
    def bulk(self, value):
        raise AttributeError('Properties cannot be directly set to Components. Try to modify the corresponding ElasticPropertySet, Instead')
    
    @property
    def shear(self):
        """Shear modulus (GPa) taken from the component property set."""
        return rparray(self.properties.shear)
    @shear.setter
    def shear(self, value):
        raise AttributeError('Properties cannot be directly set to Components. Try to modify the corresponding ElasticPropertySet, Instead')
    
    @property
    def density(self):
        """Density (g/cc) taken from the component property set."""
        return rparray(self.properties.density)
    @density.setter
    def density(self, value):
        raise AttributeError('Properties cannot be directly set to Components. Try to modify the corresponding ElasticPropertySet, Instead')

    @property
    def p_velocity(self):
        """Effective P-wave velocity (m/s) derived from mixture moduli and density."""
        return conversions.modulus_to_pvelocity(self.bulk, self.shear, self.density)
    
    @property
    def s_velocity(self):
        """Effective S-wave velocity (m/s) derived from mixture shear and density."""
        return conversions.modulus_to_svelocity(self.shear, self.density)

    @property
    def color(self):
        """RGB color tuple from the property set."""
        return self.properties.color        
    
    # def __str__(self):
    #     return ('({}: Rho={:.3f}, K={:.2f}, G={:.2f})'.format(self.type, self.density[0], self.bulk[0], self.shear[0]))
    
    def __repr__(self):
        return ('({}: Rho={:.3f}, K={:.2f}, G={:.2f})'.format(self.type, self.density[0], self.bulk[0], self.shear[0]))

#-------------------------------------------------------------------------------
class Mineral(Component):
    """
    Mineral end-member with optional inclusion set.

    Parameters
    ----------
    property_set : ElasticPropertySet
        Elastic properties for the mineral.
    porosity_weight : float, optional
        Weight used when distributing porosity among minerals (default 1.0).

    Attributes
    ----------
    inclusion_set : dict
        Dictionary mapping Inclusion objects to fractional occupancy (internal usage).
    porosity_weight : float
        Weight used for mineral-specific effective porosity allocation.
    """
    def __init__(   self, 
                    property_set: ElasticPropertySet,
                    porosity_weight = 1.) -> None:
        super().__init__(property_set)
        self.inclusion_set = {}
        self.porosity_weight = porosity_weight

    @property
    def greenberg_castagna_coefs(self):
        """Placeholder for Greenberg-Castagna coefficients (if assigned)."""
        return self._gc_coefs
    @greenberg_castagna_coefs.setter
    def greenberg_castagna_coef(self, value):
        self._gc_coef = value

    def add_inclusion(self, inclusion: Inclusion, fraction: float = 1):
        """Assign an Inclusion to this mineral with a given fraction in-host."""
        self.inclusion_set[inclusion] = fraction

#-------------------------------------------------------------------------------
class Fluid(Component):
    """
    Fluid component wrapper.

    Parameters
    ----------
    property_set : ElasticPropertySet
        Elastic property description for the fluid (bulk modulus, density).
    """
    def __init__( self, 
                  property_set: ElasticPropertySet) -> None:
        super().__init__(property_set)

#===============================================================================
class ComponentSet(Material):
    """
    A mixture of Components (minerals or fluids) with a specified mixing rule.

    The ComponentSet manages component proportions (may be un-normalized
    "proportions") and computes mixture (effective) properties (bulk/shear/density)
    using a MixingModel selected by mixing_method.

    Parameters
    ----------
    component_set : dict
        Mapping Component -> proportion (array-like). Proportions are normalized
        internally to `fractions` for property calculations.
    mixing_method : BoundMethods.MixingMethodName
        Mixing rule name used by the MixingModel (e.g., Hashin-Shtrikman, VRH).
    upper_weight : float, optional
        Weight of upper bound in the mixing models (default 0.5).
    color : tuple, optional
        RGB color used for visualization.

    Notes
    -----
    - Use set_proportions / set_component_proportion to update composition.
    """
    def __init__( self, 
                  component_set: Dict[Component, np.ndarray],
                  mixing_method: BoundMethods.MixingMethodName | Literal['voigt_reuss_hill', 'hashin_shtrikman_walpole'],
                  upper_weight: float = 0.5 ) -> None:
        self.type = MaterialType.ComponentSet
        self.component_set = component_set
        self.mixing_method = mixing_method
        self.upper_weight = upper_weight
        self.id = random.getrandbits(128)
        self.color = Color()

    @property
    def mixing_model(self) -> MixingModel:
        """Return a MixingModel instance configured with the selected mixing method."""
        return MixingModel(self.mixing_method, self.upper_weight)

    @property
    def components(self) -> List[Component]:
        """List of Component keys in the internal component_set dictionary."""
        self._components = list(self.component_set.keys())
        return self._components

    @property
    def component_set(self) -> Dict[Component, np.ndarray]:
        """Mapping Component -> proportion (array-like stored as numpy arrays)."""
        return self._component_set
    @component_set.setter
    def component_set(self, value):
        self._component_set = {}
        if isinstance(value, Component):
            value = {value: rparray(1)}
        for component, proportion in value.items():
            self._component_set[component] = rparray(proportion)

    @property
    def fraction_set(self) -> Dict[Component, np.ndarray]:
        """Normalized fractions (Component -> fraction) computed from proportions."""
        fraction_set = {}
        for component in self.components:
            fraction_set[component] = self.component_set[component]/self.total_proportion
        return fraction_set

    @property
    def bulk(self):
        """Effective bulk modulus in GPa computed from the mixing model."""
        bulk, _ = self.moduli()
        return bulk
    @bulk.setter
    def bulk(self, value):
        """Bulk modulus is derived for ComponentSet and cannot be directly set."""
        raise AttributeError('Properties cannot be directly set for {0}. Try to define a new {0} with a single component and assign the values to the component properties'.format(type(self).__name__))
  
    @property
    def shear(self):
        """Effective shear modulus in GPa computed from the mixing model."""
        _, shear = self.moduli()
        return shear        
    @shear.setter
    def shear(self, value):
        """Shear modulus is derived and cannot be directly set."""
        raise AttributeError('Properties cannot be directly set for {0}. Try to define a new {0} with a single component and assign the values to the component properties'.format(type(self).__name__))
    
    @property
    def density(self):
        """Effective density in g/cc."""
        fractions = self.fractions
        density = DensityModel.average_density(self.densities, fractions)
        return density
    @density.setter
    def density(self, value):
        """Density for ComponentSet cannot be directly set; set component densities instead."""
        raise AttributeError('Properties cannot be directly set for {0}. Try to define a new {0} with a single component and assign the values to the component properties'.format(type(self).__name__))

    @property
    def p_velocity(self):
        """Effective P-wave velocity (m/s) derived from mixture moduli and density."""
        return conversions.modulus_to_pvelocity(self.bulk, self.shear, self.density)
    @p_velocity.setter
    def p_velocity(self, value):
        raise AttributeError('Properties cannot be directly set for {0}. Try to define a new {0} with a single component and assign the values to the component properties'.format(type(self).__name__))
    
    @property
    def s_velocity(self):
        """Effective S-wave velocity (m/s) derived from mixture shear and density."""
        return conversions.modulus_to_svelocity(self.shear, self.density)
    @s_velocity.setter
    def s_velocity(self, value):
        raise AttributeError('Properties cannot be directly set for {0}. Try to define a new {0} with a single component and assign the values to the component properties'.format(type(self).__name__))

    @property
    def bulks(self) -> List[np.ndarray]:
        """List of component bulk moduli (GPa)."""
        bulks = [component.bulk for component in self.component_set.keys()]
        return bulks
    
    @property
    def shears(self) -> List[np.ndarray]:
        """List of component shear moduli (GPa)."""
        shears = [component.shear for component in self.component_set.keys()]
        return shears
    
    @property
    def densities(self) -> List[np.ndarray]:
        """List of component densities (g/cc)."""
        densities = [component.density for component in self.component_set.keys()]
        return densities

    @property
    def components_count(self) -> int:
        """Number of components in the set."""
        return len(self.component_set)

    @property
    def proportions(self) -> List[np.ndarray]:
        """List of raw proportions (may not sum to 1)."""
        return list(self.component_set.values())

    @property
    def fractions(self) -> List[np.ndarray]:
        """List of normalized fractions summing to 1 across components."""
        return list(self.fraction_set.values())

    @property
    def stacked_fractions(self) -> np.ndarray:
        """Cumulative sum of fractions along the first axis for mixture sampling/stacking."""
        return np.cumsum(self.fractions, axis=0)

    @property
    def total_proportion(self) -> np.ndarray:
        """Total (unnormalized) proportion obtained by summing component proportions."""
        total_value = 0
        for proportion in self.proportions:
            total_value += proportion
        return total_value

    def component_fraction(self, component: Component) -> np.ndarray:
        """Return the normalized fraction array for a given component."""
        return self.fraction_set[component]
    
    def get_component_by_type(self, type:str|Component) -> Component:
        """Return the first component matching the provided type string (case-insensitive)."""
        for component in self.components:
            if component.properties.type.lower() == type.lower():
                return component

    def add_component(self, component: Component, fraction: np.ndarray, isproportion = False) -> None:
        """
        Add or update a component with a provided fraction/proportion.

        Parameters
        ----------
        component : Component
            Component instance to add.
        fraction : array-like
            Fraction (if isproportion=False) or proportion (if isproportion=True).
        isproportion : bool, optional
            If True, the provided value is a raw proportion; otherwise it is treated
            as a desired normalized fraction and converted to the internal proportion
            representation.
        """
        if isproportion: # Given value is a proportion (un-normalized) value
            coef = 1
        else: # Given value is a true (normalized) fraction value
            #The coefficient is calculated such that the provided fraction be the final (normalized) fraction.
            coef = self.total_proportion/(1-fraction)
        self.component_set[component] = fraction*coef
        
    def remove_component(self, component: Component) -> None:
        """Remove a component from the set."""
        self.component_set.pop(component)

    def set_component_proportion(self, component: Component, proportion: np.ndarray):
        """Set the internal (possibly un-normalized) proportion for a component."""
        self.component_set[component] = rparray(proportion)
    
    def set_proportions(self, proportion_list: List[np.ndarray]) -> None:
        """Set proportions for all components at once (order must match self.components)."""
        #This function assumes that order of component in the fraction_list is similar to order of components in self.component_set
        if len(proportion_list) == self.components_count:
            for  idx, component in enumerate(self.components):
                self.set_component_proportion(component, proportion_list[idx])
        else:
            ValueError(f"Expected {self.components_count} fractions, got {len(proportion_list)}")

    def moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the effective bulk and shear moduli for the mixture using the mixing model.

        Returns
        -------
        (bulk, shear) : tuple
            Mixture bulk and shear moduli (GPa).
        """
        bulk, shear = self.mixing_model.method(self.bulks, self.shears,  self.fractions, self.upper_weight)
        return bulk, shear 

    def simulate(self, proportion_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate mixture velocities and density for a candidate proportion list.

        Returns computed p_velocity (m/s), s_velocity (m/s), and density (g/cc).
        """
        componentset = deepcopy(self)
        componentset.set_proportions(proportion_list)
        bulk, shear = componentset.moduli()
        density = componentset.density
        pvelocity, svelocity = conversions.modulus_to_velocity(bulk, shear, density)
        return pvelocity, svelocity, density

    def __str__(self):
        components = self.components
        text = '{:20s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|{:^15s}\n'.format('Type','Fraction', 'Density', 'P-velocity', 'S-velocity', 'Bulk', 'Shear', 'Poisson')
        text += '-'*130 + '\n'
        for item in components:
            text += '{:20s}|{:^15.4f}|{:^15.3f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.3f}\n'.format(item.type, 
                                                                                                            self.component_fraction(item).mean(),
                                                                                                            item.density.mean(), 
                                                                                                            item.p_velocity.mean(), 
                                                                                                            item.s_velocity.mean(),
                                                                                                            item.bulk.mean(),
                                                                                                            item.shear.mean(),
                                                                                                            item.poisson.mean())
        text += '-'*130 + '\n'
        text += '{:20s}|{:^15.3f}|{:^15.3f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.2f}|{:^15.3f}\n'.format(type(self).__name__, 
                                                                                                            1.00,
                                                                                                            self.density.mean(), 
                                                                                                            self.p_velocity.mean(), 
                                                                                                            self.s_velocity.mean(),
                                                                                                            self.bulk.mean(),
                                                                                                            self.shear.mean(),
                                                                                                            self.poisson.mean())
        return text
    
    def __repr__(self):
        text= ''
        for component, fraction in self.component_set.items():
            text += '{}: {:.3f}\n'.format(component.type, fraction)

        return '{}'.format(text)

#-------------------------------------------------------------------------------
class MineralSet(ComponentSet):
    """
    Specialized ComponentSet for minerals.

    Provides helpers for mineral-specific porosity partitioning and volume fractions.
    """
    def __init__( self, 
                  mineral_set: Dict[Mineral, np.ndarray],
                  mixing_method: BoundMethods | Literal['voigt_reuss_hill', 'hashin_shtrikman_walpole'] = BoundMethods.MixingMethodName.HashinShtrikmanWalpole,
                  upper_weight: float = 0.5 ) -> None:
        super().__init__(mineral_set, mixing_method, upper_weight)

    @property
    def minerals(self) -> List[Mineral]:
        """List of mineral components."""
        return list(self.components)

    def volume_fraction(self, mineral: Mineral) -> List[np.ndarray]:
        """Return the volume fraction array for the specified mineral."""
        if mineral in self.component_set:
            return self.component_fraction(mineral)
        else:
            raise ValueError('Mineral does not exist.')

    def set_porosity_weight(self, mineral: Mineral, weight: float) -> None:
        """Set mineral-specific porosity weight used by mineral_porosity_fraction."""
        if mineral in self.components:
            mineral.porosity_weight = weight

    def mineral_porosity_fraction(self, mineral: Material) -> np.ndarray:
        """Portion of total porosity assigned to the given mineral (dimensionless)."""
        return self.fraction_set[mineral] * mineral.porosity_weight/self._porosity_fraction_normalizer

    def porosity_weights_set(self):
        """Return a dictionary of porosity weights corresponding to each mineral.
        """
        weight_set = {}
        for mineral in self.minerals:
            weight_set[mineral] = mineral.porosity_weight
        return weight_set

    @property
    def _porosity_fraction_normalizer(self) -> np.ndarray:
        """Internal normalizer summing porosity weights across minerals."""
        total_weight = 0
        for mineral in self.minerals:
            total_weight += mineral.porosity_weight*self.fraction_set[mineral]
        return total_weight

#-------------------------------------------------------------------------------
class FluidSet(ComponentSet):
    """
    ComponentSet specialized for fluids.

    FluidSet supports the notion of a bound (immobile) fluid (e.g., wetting film)
    that occupies a fraction of the total porosity and is treated separately
    in certain effective-medium calculations.
    """
    def __init__( self, 
                  fluid_set: Dict[Fluid, np.ndarray],
                  reuss_weight: float = 1. ) -> None:
        super().__init__(fluid_set, BoundMethods.MixingMethodName.VoigtReussHill, 1-reuss_weight)
        self.bound_fluid = self.get_brine() #Rock is assumed to be water-wet by default but it may be changed if needed.

    @property
    def fluids(self) -> List[Fluid]:
        """List of fluid components."""
        return list(self.components)

    @property
    def bound_fluid(self) -> Fluid:
        """Bound (wetting) fluid component used in effective porosity conversions."""
        return self._bound_fluid
    @bound_fluid.setter
    def bound_fluid(self, value:Fluid):
        if value in self.component_set:
            self._bound_fluid = value
        else:
            raise ValueError("Given fluid does not belong to this FluidSet")


    def get_brine(self) -> Fluid:
        """Convenience: return first component matching FluidType.Brine (if present)."""
        return self.get_component_by_type(FluidType.Brine)

    def saturation(self, fluid: Fluid) -> List[np.ndarray]:
        """Return saturation (fraction) array for a given fluid component."""
        self.component_fraction(fluid)
    





class Rock(Material):
    """
    Abstract base class for a rock model combining a MineralSet and a FluidSet.

    Rock manages porosity, effective porosity, bound-fluid porosity and delegates
    frame and fluid-effect calculations to model classes (Inclusion, Contact or
    granular methods provided by models/effective_medium).

    Parameters
    ----------
    mineralset : MineralSet
        Mineral mixture describing the rock frame.
    fluidset : FluidSet
        Fluid mixture occupying pore space.
    total_porosity : array-like or float
        Total porosity (fraction, 0-1).
    rock_frame_method : enum or None
        Identifier selecting the rock-frame modelling approach (Inclusion/Contact).
    critical_porosity : float, optional
        Critical porosity used by certain inclusion/contact models.
    bound_fluid_porosity : float, optional
        Portion of total porosity occupied by bound (immobile) fluid.
    analysis_frequency : float, optional
        Frequency (Hz) used for frequency-dependent fluid effect calculations.
    """
    def __init__(self, 
                 mineralset: MineralSet, 
                 fluidset: FluidSet, 
                 total_porosity,  
                 rock_frame_method: None | BoundMethods.RockMethodName | InclusionMethods.InclusionMethodName | ContactMethods.ContactMethodName, 
                 critical_porosity = 1,
                 bound_fluid_porosity = 0,
                 analysis_frequency = 1):
        self.mineralset = mineralset
        self.fluidset = fluidset
        self.effective_method = rock_frame_method
        self.total_porosity = total_porosity
        self.critical_porosity = critical_porosity
        self.bound_fluid_porosity = bound_fluid_porosity
        self.id = random.getrandbits(128)
        self.type = MaterialType.Rock
        self.viscosity_to_permeability = 0.4 #cp/md
        self.porosity_weight = 1.
        self.analysis_frequency = analysis_frequency
        self.Z = 0.003 #This is the Fluid-effect tuning parameter (Dvorkin et all, 1995)
        self.color = Color()

    @property
    def fluidset(self) -> FluidSet:
        """FluidSet associated with the rock. If a single Fluid is provided it is coerced to a FluidSet."""
        return self._fluidset
    @fluidset.setter
    def fluidset(self, value):
        if not isinstance(value, FluidSet):
            value = FluidSet({value: 1})
        self._fluidset = value

    @property
    def dry_fluidset(self) -> FluidSet:
        """
        Construct a 'dry' FluidSet where the non-bound pore space is occupied by a
        Dry fluid (used to model dry-frame states).
        """
        dry = Fluid(FluidsPropertyTable().Dry)
        dry_fluidset = deepcopy(self.fluidset)
        dry_fluidset.component_set = {self.fluidset.bound_fluid: self.bound_fluid_porosity/self.total_porosity, dry:self.effective_porosity/self.total_porosity}
        return dry_fluidset

    @property
    def mineralset(self) -> MineralSet:
        """MineralSet associated with the rock. Single minerals are coerced to a MineralSet."""
        return self._mineralset
    @mineralset.setter
    def mineralset(self, value):
        if not isinstance(value, MineralSet):
            value = MineralSet({value: 1})
        self._mineralset = value

    @property
    def minerals(self) -> List[Mineral]:
        """List of Mineral components in the mineralset."""
        return self.mineralset.components

    @property
    def fluids(self) -> List[Fluid]:
        """List of Fluid components in the fluidset."""
        return self.fluidset.components

    @property
    def effective_fluidset(self) -> FluidSet:
        """FluidSet remapped from total-porosity to effective-porosity basis (uses conversions.total_to_effective_fluidset)."""
        return conversions.total_to_effective_fluidset(self.fluidset, self.total_porosity, self.effective_porosity)

    @property
    def dry_rock(self):
        """Return a deepcopy of the rock where the pore space is replaced with the dry fluidset."""
        dry_rock = deepcopy(self)
        dry_rock.fluidset = self.dry_fluidset
        return dry_rock

    @property
    def total_porosity(self):
        """Total porosity (fraction, 0-1)."""
        return self._porosity
    @total_porosity.setter
    def total_porosity(self, value):
        self._porosity = rparray(value)

    @property
    def effective_porosity(self):
        """Effective porosity = total_porosity - bound_fluid_porosity (fraction)."""
        return self.total_porosity - self.bound_fluid_porosity# + np.finfo(np.float32).eps

    @property
    def bound_fluid_porosity(self):
        """Porosity occupied by bound (immobile) fluid (fraction)."""
        return self._bound_porosity
    @bound_fluid_porosity.setter
    def bound_fluid_porosity(self, value):
        self._bound_porosity = rparray(value)

    def mineral_porosity(self, mineral: Mineral) -> np.ndarray:
        """Return porosity assigned to a particular mineral host (array of effective porosity fractions)."""
        return self.total_porosity * self.mineralset.mineral_porosity_fraction(mineral)

    def minerals_porosity_set(self):
        porosity_dict = {}
        for mineral in self.minerals:
            porosity_dict[mineral] = self.mineral_porosity(mineral)
        return porosity_dict

    @property
    def bulk(self):
        """Rock saturated bulk modulus in GPa (computed by moduli())."""
        bulk, _ = self.moduli()
        return bulk
    @bulk.setter
    def bulk(self, value):
        """Allow explicit override of rock bulk modulus (rare; generally computed)."""
        self._bulk = value
  
    @property
    def shear(self):
        """Rock shear modulus in GPa (computed by moduli())."""
        _, shear = self.moduli()
        return shear
    @shear.setter
    def shear(self, value):
        """Allow explicit override of rock shear modulus."""
        self._shear = value
    
    @property
    def density(self):
        """Bulk rock density (g/cc) computed as pore-weighted average of minerals and fluids."""
        self._density = DensityModel.average_density([self.mineralset.density, self.fluidset.density], [1-self.total_porosity, self.total_porosity])
        return self._density
    @density.setter
    def density(self, value):
        """Set rock density explicitly (overrides computed average)."""
        self._density = value

    @property
    def p_velocity(self):
        """Effective P-wave velocity (m/s) derived from mixture moduli and density."""
        return conversions.modulus_to_pvelocity(self.bulk, self.shear, self.density)
    
    @property
    def s_velocity(self):
        """Effective S-wave velocity (m/s) derived from mixture shear and density."""
        return conversions.modulus_to_svelocity(self.shear, self.density)

    @property
    def viscosity_to_permeability(self):
        """Viscosity-to-permeability ratio used for Biot-frequency estimation (user units)."""
        return self._eta_to_k
    @viscosity_to_permeability.setter
    def viscosity_to_permeability(self, value):
        self._eta_to_k = value

    @property
    def biot_characteristic_frequency(self):
        """Biot characteristic frequency (Hz) separating low/high frequency fluid effects."""
        return biot_characteristic_frequency(self.total_porosity, self.fluidset.density, self.viscosity_to_permeability)

    @property
    def Z(self) -> float:
        """Dvorkin fluid effect model's tuning factor."""
        return self._Z
    @Z.setter
    def Z(self, value):
        self._Z = value

    def wetsolid_moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute moduli of the wet solid frame (override in subclasses)."""
        RuntimeWarning("Rock frame is not implemented. Use a specified rock model such as Inclusion or Granular.")

    def frame_moduli(self):
        """Compute dry-frame moduli (override in subclasses)."""
        RuntimeWarning("Rock frame is not implemented. Use a specified rock model such as Inclusion or Granular.")

    def hiPressure_dry_moduli(self):
        """Compute dry-frame moduli at high confining pressure (override in subclasses)."""
        RuntimeWarning("Rock frame is not implemented. Use a specified rock model such as Inclusion or Granular.")
    
    def set_volume_fractions(self, volume_fraction_list: List[np.ndarray]) -> None:
        """Set mineral volume fractions via the mineralset helper."""
        self.mineralset.set_component_proportion(volume_fraction_list)

    def set_saturations(self, saturation_list: List[np.ndarray]) -> None:
        """Set fluid saturations for the fluidset."""
        self.fluidset.set_component_proportion(saturation_list)

    def set_reservoir_properties(self, total_porosity: np.ndarray, volume_fraction_list: List[np.ndarray], saturation_list: List[np.ndarray]):
        """Convenience: set total porosity, mineral fractions and fluid saturations together."""
        self.total_porosity = total_porosity
        self.set_volume_fractions(volume_fraction_list)
        self.set_saturations(saturation_list)

    def set_total_porosity(self, total_porosity: np.ndarray):
        """Set total porosity."""
        self.total_porosity = total_porosity

    def set_volume_fractions(self, fraction_list: List[np.ndarray]):
        """Set mineral volume fractions (alias)."""
        self.mineralset.set_proportions(fraction_list)

    def set_saturations(self, saturation_list: List[np.ndarray]):
        """Set fluid saturations (alias)."""
        self.fluidset.set_proportions(saturation_list)
    
    def set_reservoir_properties(self, total_porosity: np.ndarray, volume_fraction_list: List[np.ndarray], saturation_list: List[np.ndarray]):
        """(Duplicate alias) Set full reservoir properties on the rock."""
        self.total_porosity = total_porosity
        self.set_volume_fractions(volume_fraction_list)
        self.set_saturations(saturation_list)

    def simulate(self, total_porosity: np.ndarray, volume_fraction_list: List[np.ndarray], saturation_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate rock velocities and density for a candidate reservoir state.

        Parameters
        ----------
        total_porosity : array-like
            Total porosity for the simulated state.
        volume_fraction_list : list of array-like
            Mineral fractions for the simulated state.
        saturation_list : list of array-like
            Fluid saturations for the simulated state.

        Returns
        -------
        (p_velocity, s_velocity, density) : tuple
            Simulated P- and S-wave velocities (m/s) and bulk density (g/cc).
        """
        rock = deepcopy(self) #Copy the object to keep its state unchanged.
        rock.set_reservoir_properties(total_porosity, volume_fraction_list, saturation_list)
        bulk, shear = rock.moduli()
        density = rock.density
        pvelocity, svelocity = conversions.modulus_to_velocity(bulk, shear, density)
        return pvelocity, svelocity, density


#-------------------------------------------------------------------------------
class SimpleRock(Rock):
    """
    Simple rock object created directly from specified velocities.

    SimpleRock allows representing an observed rock by directly setting measured
    p_velocity and s_velocity and computing bulk/shear moduli from those values.
    """
    def __init__(self, 
                 mineralset: MineralSet, 
                 fluidset: FluidSet,
                 total_porosity,
                 p_velocity: np.ndarray,
                 s_velocity: np.ndarray): 
        super().__init__(mineralset, fluidset, total_porosity, rock_frame_method = None, critical_porosity=1.)
        self.bulk = conversions.velocity_to_bulk(p_velocity, s_velocity, self.density)
        self.shear = conversions.velocity_to_shear(s_velocity, self.density)

    @property
    def bulk(self):
        """Rock saturated bulk modulus in GPa (computed by moduli())."""
        return self._bulk
    @bulk.setter
    def bulk(self, value):
        """Allow explicit override of rock bulk modulus (rare; generally computed)."""
        self._bulk = value
  
    @property
    def shear(self):
        """Rock shear modulus in GPa (computed by moduli())."""
        return self._shear
    @shear.setter
    def shear(self, value):
        """Allow explicit override of rock shear modulus."""
        self._shear = value

#-------------------------------------------------------------------------------
class InclusionRock(Rock):
    """
    Rock model that represents pores and grains as inclusions (inclusion-based effective medium).

    Inclusions are objects (effective_medium.Inclusion) with content, geometry and
    mechanical contrast. The InclusionRock delegates wet/dry frame calculations to
    an InclusionModel configured by effective_method.

    Parameters
    ----------
    mineralset : MineralSet
    fluidset : FluidSet
    total_porosity : array-like
    rock_frame_method : InclusionMethods.InclusionMethodName, optional
        Inclusion-based effective-medium method (e.g., DEM).
    type : str, optional
        Descriptive label for the rock.
    critical_porosity : float, optional
        Critical porosity used by the inclusion model.
    """
    def __init__(self, 
                 mineralset: MineralSet, 
                 fluidset: FluidSet, 
                 total_porosity: np.ndarray,
                 rock_frame_method: InclusionMethods.InclusionMethodName | Literal['mori_tanaka','kuster_toksoz','self_consistent', 'differential_effective_medium','dem_modified', 'mt', 'kt', 'sc', 'dem'] = InclusionMethods.InclusionMethodName.DEM):
        super().__init__(mineralset, fluidset, total_porosity, rock_frame_method)
        self._inclusions = []

    @property
    def rock_frame_model(self) -> InclusionModel:
        """Return an InclusionModel instance configured for this rock's method and critical porosity."""
        return InclusionModel(self.effective_method)

    @property
    def inclusions(self) -> List[Inclusion]:
        """List of Inclusion objects associated with the rock."""
        return self._inclusions

    @property
    def inclusion_set(self) -> Dict[Inclusion, np.ndarray]:
        """Mapping Inclusion -> inclusion porosity (absolute porosity assigned to each inclusion)."""
        normalized_set = {}
        for inclusion in self.inclusions:
            normalized_set[inclusion] = self.inclusion_porosity(inclusion)
        return normalized_set

    @property
    def bound_inclusions(self) -> List[Inclusion]:
        """List of inclusions whose content matches the bound fluid (wetting phase)."""
        return [inclusion for inclusion in self.inclusions if inclusion.content == self.fluidset.bound_fluid.properties]

    @property
    def nonbound_inclusions(self) -> List[Inclusion]:
        """List of inclusions that represent non-bound (free) pores."""
        return [inclusion for inclusion in self.inclusions if inclusion not in self.bound_inclusions]

    @property
    def stiff_inclusions(self) -> List[Inclusion]:
        """List of mechanically stiff inclusions (inclusion.isstiff True)."""
        return [inclusion for inclusion in self.inclusions if inclusion.isstiff]

    @property
    def dry_inclusion_set(self) -> Dict[Inclusion, np.ndarray]:
        """Return an inclusion set where all fluids are replaced by a 'Dry' fluid placeholder."""
        dry_inclusion_set = {}
        for inclusion in self.inclusion_set:
            dry_inclusion = deepcopy(inclusion)
            dry_inclusion.content = FluidsPropertyTable().Dry
            dry_inclusion_set[dry_inclusion] = self.inclusion_set[inclusion]
        return dry_inclusion_set
   
    @property
    def bound_inclusion_set(self) -> Dict[Inclusion, np.ndarray]:
        """Inclusion set containing only bound (wet) inclusions and their porosities."""
        wet_inclusion_set = {}
        for inclusion in self.inclusions:
            if inclusion.content == self.fluidset.bound_fluid.properties:
                wet_inclusion_set[inclusion] = self.inclusion_set[inclusion]
        return wet_inclusion_set

    @property
    def nonbound_inclusion_set(self) -> Dict[Inclusion, np.ndarray]:
        """Inclusion set containing only non-bound inclusions and their porosities."""
        nonbound_inclusion_set = {}
        for inclusion in self.inclusion_set:
            if not inclusion.content == self.fluidset.bound_fluid.properties:
                nonbound_inclusion_set[inclusion] = self.inclusion_set[inclusion]
        return nonbound_inclusion_set
  
    @property
    def stiff_inclusion_set(self) -> Dict[Inclusion, np.ndarray]:
        """Inclusion set containing only stiff inclusions and their porosities."""
        stiff_inclusion_set = {}
        for inclusion in self.inclusions:
            if inclusion.isstiff:
                stiff_inclusion_set[inclusion] = self.inclusion_set[inclusion]
        return stiff_inclusion_set

    @property
    def bound_fluid_porosity(self):
        """Compute the porosity occupied by bound inclusions (sum of bound inclusion porosities)."""
        bound_fluid_porosity = 0
        for inclusion in self.bound_inclusions:
                bound_fluid_porosity += self.inclusion_set[inclusion]
        return bound_fluid_porosity
    @bound_fluid_porosity.setter
    def bound_fluid_porosity(self, value):
        pass    #PASS, because in Inclusion Rocks bound porosity is not stored, but is calculated from bound inclusions.
    
    @property
    def _total_porosity_weight(self) -> np.ndarray:
        """Sum of host-weighted inclusion fractions used for normalizing inclusion assignments."""
        total_weight = 0
        for inclusion in self.inclusions:
            total_weight += inclusion.fraction_in_host * self.mineralset.mineral_porosity_fraction(inclusion.host)
        return total_weight

    def add_inclusion(self, inclusion: Inclusion, host: Material, fraction_in_host: float = 1.0) -> None:
        """Attach an Inclusion to the rock and assign it to a mineral host."""
        if isinstance(inclusion, Inclusion) and host in self.minerals:
            inclusion.host = host
            inclusion.fraction_in_host = fraction_in_host
            self._inclusions.append(inclusion)

    def remove_inclusion(self, inclusion: Inclusion) -> None:
        """Remove an Inclusion from the rock."""
        self._inclusions.remove(inclusion)

    def clear_inclusions(self) -> None:
        """Remove all inclusions from the rock (clear list)."""
        self.inclusions = []

    def inclusion_porosity(self, inclusion: Inclusion) -> np.ndarray:
        """Absolute porosity assigned to a specific inclusion (total_porosity * inclusion fraction)."""
        inclusion_porosity = self.total_porosity * self.inclusion_porosity_fraction(inclusion)
        return inclusion_porosity

    def inclusion_porosity_fraction(self, inclusion : Inclusion) -> np.ndarray:
        """
        Fraction of the rock's total porosity assigned to the given inclusion type.

        The fraction is proportional to the inclusion's fraction_in_host and the
        hosting mineral's porosity allocation.
        """
        return inclusion.fraction_in_host * self.mineralset.mineral_porosity_fraction(inclusion.host)/self._total_porosity_weight

    def host_inclusion_list(self, host: Material) -> List[Inclusion]:
        """
        Return inclusions hosted by a given mineral and warn if the host fractions don't sum to unity.
        """
        inclusions_in_host = []
        total_fraction_in_host = 0
        for inclusion in self.inclusions:
            if host == inclusion.host:
                inclusions_in_host.append(inclusion)
                total_fraction_in_host += inclusion.fraction_in_host
        if np.any(total_fraction_in_host) != 1:
            Warning("Total fractions in this host does not add-up to unity for some samples.")
        return inclusions_in_host

    def hiPressure_dry_moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dry-frame moduli computed at high confining pressure (GPa)."""
        dry_bulk_hiP, dry_shear_hiP = self.rock_frame_model.method(self.mineralset.bulk, self.mineralset.shear, self.stiff_inclusion_set)
        return dry_bulk_hiP, dry_shear_hiP

    def wetsolid_moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wet-solid moduli (inclusions filled with bound fluid) computed by the inclusion model."""
        bulk, shear = self.rock_frame_model.method(self.mineralset.bulk, self.mineralset.shear, self.bound_inclusion_set)
        return bulk, shear

    def frame_moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dry-frame moduli computed by embedding non-bound inclusions into the wet-solid frame."""
        wetsolid_bulk, wetsolid_shear = self.wetsolid_moduli()
        dry_bulk, dry_shear = self.rock_frame_model.method(wetsolid_bulk, wetsolid_shear, self.nonbound_inclusion_set)
        return dry_bulk, dry_shear

    def moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute saturated rock moduli (bulk, shear) accounting for fluid effects.

        Process:
        - Compute wetsolid moduli (bound inclusions present).
        - Compute dry-frame moduli by adding non-bound inclusions.
        - Optionally compute high-pressure dry-frame moduli for frequency-dependent models.
        - Apply frequency-dependent fluid effect model (Dvorkin et al., 1995) if non-bound inclusions exist.

        Returns
        -------
        (bulk, shear) : tuple
            Saturated rock bulk and shear moduli (GPa).
        """
        #We use frequency dependant fluid effect model of Dvorkin et al (1995). This model in low frequency (f=0) is identical to Gassman
        wetsolid_bulk, _ = self.wetsolid_moduli()
        frame_bulk, frame_shear = self.frame_moduli()
        dry_bulk_hiP, _ = self.hiPressure_dry_moduli()
        if self.nonbound_inclusion_set:
            bulk, shear = FluidEffectMethods.dvorkin(frame_bulk, frame_shear, wetsolid_bulk, self.effective_fluidset.bulk, self.effective_porosity, self.analysis_frequency, self.mineralset.density, self.effective_fluidset.density, dry_bulk_hiP, self.Z)
        else:
            bulk, shear = frame_bulk, frame_shear
        return bulk, shear 

#-------------------------------------------------------------------------------
class GranularRock(Rock):
    """
    Granular/contact-based rock model (Hertz-Mindlin and contact cementing variants).

    GranularRock uses ContactModel to compute contact-frame dry moduli based on
    coordination number, adhesion coefficient and confining pressure. Final saturated
    moduli are computed by applying a fluid-effect model (Dvorkin) to the dry frame.

    Parameters
    ----------
    mineralset : MineralSet
    fluidset : FluidSet
    total_porosity : array-like
    rock_frame_method : ContactMethods.ContactMethodName
        Contact-based method identifier.
    contact_no : int, optional
        Coordination number (average contacts per grain).
    adhesion_coef : float, optional
        Tangential adhesion coefficient affecting contact stiffness.
    critical_porosity : float, optional
        Uncemented porosity threshold used by contact methods.
    cement : Material, optional
        Cement material used when modeling cemented contacts.
    pressure : float, optional
        Confining/contact pressure (Pa or consistent units).
    contact_cement_saturation : float or array-like, optional
        Fraction of contacts cemented (0-1).
    """
    def __init__(self, 
                 mineralset: MineralSet, 
                 fluidset: FluidSet, 
                 total_porosity: np.ndarray,  
                 rock_frame_method: ContactMethods.ContactMethodName | Literal['hertz_mindlin','soft_sand','stiff_sand','intermediate_sand','intermediate_stiff_sand','intermediate_cemented_sand','contact_cemented_sand','surface_cemented_sand','digby','jenkins','walton_hydrostatic_rough','walton_hydrostatic_smooth', 'scement', 'ccement', 'midcemented', 'midstiff', 'mid', 'stiff', 'soft', 'hm'], 
                 contact_no: float = 9, 
                 adhesion_coef: float = 1, 
                 critical_porosity: float = None, 
                 cement: Material = None, 
                 pressure: np.ndarray = None, 
                 contact_cement_saturation: np.ndarray = None):
        super().__init__(mineralset, fluidset, total_porosity, rock_frame_method, critical_porosity)
        self.contact_no = contact_no
        self.adhesion_coef = adhesion_coef
        self.critical_porosity = critical_porosity
        self.pressure = pressure
        self.cement = cement
        self.contact_cement_saturation = contact_cement_saturation
        self.hi_pressure = 100e6


    @property
    def contact_cement_saturation(self):
        """Fraction of contacts that are cemented (0-1)."""
        return self._contact_cement_saturation
    @contact_cement_saturation.setter
    def contact_cement_saturation(self, value):
        self._contact_cement_saturation = value

    @property
    def cement_material(self):
        """Cement material used by contact models (Material or None)."""
        return self._cement
    @cement_material.setter
    def cement_material(self, value: Material):
        self._cement = value

    @property
    def contact_no(self):
        """Coordination number (average contacts per grain)."""
        return self._contact_no
    @contact_no.setter
    def contact_no(self, value):
        self._contact_no = value

    @property
    def rock_frame_model(self) -> ContactModel:
        """Return a ContactModel configured with current contact parameters."""
        return ContactModel(self.effective_method, adhesion_coef = self.adhesion_coef, uncemented_porosity = self.critical_porosity, pressure = self.pressure, cement = self.cement, contact_cement_saturation = self.contact_cement_saturation)

    def frame_moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute contact-frame (dry) bulk and shear moduli using the contact model."""
        dry_bulk, dry_shear = self.rock_frame_model.method(self.mineralset.bulk, self.mineralset.shear, self.total_porosity, self.contact_no)
        return dry_bulk, dry_shear

    def hiPressure_dry_moduli(self)  -> Tuple[np.ndarray, np.ndarray]:
        """Compute dry-frame moduli at a high confining pressure value (hi_pressure attribute)."""
        hi_pressure_model =  ContactModel(self.effective_method, adhesion_coef = self.adhesion_coef, uncemented_porosity = self.critical_porosity, pressure = self.hi_pressure, cement = self.cement, contact_cement_saturation = self.contact_cement_saturation)
        dry_bulk_hiP, dry_shear_hiP = hi_pressure_model.method(self.mineralset.bulk, self.mineralset.shear, self.total_porosity, self.contact_no)
        return dry_bulk_hiP, dry_shear_hiP
  
    def moduli(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute saturated rock moduli for a granular/contact rock.

        Steps:
        - compute dry-frame moduli using contact model
        - compute hi-pressure dry-frame moduli for fluid-effect model
        - apply Dvorkin frequency-dependent fluid effect to yield saturated moduli
        """
        #We use frequency dependant fluid effect model of Dvorkin et al (1995). This model in low frequency (f=0) is identical to Gassman
        #We assume in Granular Rock that total porosity is saturated at once (bound and non-bound pores are not separated for saturation)
        dry_bulk, dry_shear = self.frame_moduli()
        dry_bulk_hiP, _ = self.hiPressure_dry_moduli()
        bulk, shear = FluidEffectMethods.dvorkin(dry_bulk, dry_shear, self.mineralset.bulk, self.fluidset.bulk, self.total_porosity, self.analysis_frequency, self.mineralset.density, self.fluidset.density, dry_bulk_hiP, self.Z)
        return bulk, shear 



if __name__ == '__main__':
    pass
