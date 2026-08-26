"""
Effective-medium and empirical models

This module provides high-level model classes that wrap the core mathematical
algorithms used in rock physics modeling. It serves as the interface layer
between material properties (materials.py) and the underlying computational
methods (effective_medium.py), plus empirical regression tools for data analysis.
"""

from rokpy.avo import AVOMethods
from .constants import Color, HanTrendNames, PropertyTemplate, PropertyTemplates, VernikTrendNames, CastagnaTrendNames, GardnerTrendNames
from .utilities import MultiEnum
from .effective_medium import Inclusion, BoundMethods, InclusionMethods, ContactMethods
from enum import StrEnum
import random
import numpy as np
import scipy
from typing import Dict, List, Literal

#===================================================================================
class DensityModel():
    """Model to compute mixture densities.
    """
    def __init__(self):
        pass

    @staticmethod
    def average_density (densities: List[np.ndarray], fractions: List[np.ndarray]) -> np.ndarray:
        """Return the volumetric average of densities.

        Parameters
        ----------
        densities : list[np.ndarray]
            List/array of density arrays for each component.
        fractions : list[np.ndarray]
            List/array of volumetric fractions for each component (same shape
            as densities). Fractions are expected to sum to 1 element-wise.

        Returns
        -------
        np.ndarray
            Array of averaged densities computed element-wise.
        """
        average_density = 0
        for density, fraction in zip(densities, fractions):
            average_density += density*fraction
        return average_density

#===================================================================================
# Effective Medium Models
class EffectiveMediumModel:
    """Generic wrapper for an effective-medium method.

    This class stores a reference to the method implementation class (e.g.
    InclusionMethods, ContactMethods), an enumeration of allowed method names,
    and the currently selected method name. The `method` property returns a
    callable bound to the stored implementation class.

    Parameters
    ----------
    method_class : type
        Class containing algorithm implementations (e.g. InclusionMethods).
    method_enum : MultiEnum
        Enum type used to validate method names.
    method_name : str
        Name (or enum member) of the method to select.

    """
    def __init__(self, method_class, method_enum: MultiEnum, method_name: str):
        self.method_class = method_class
        self.method_enum = method_enum
        self.method_name = method_name
        
    @property
    def method(self) -> callable:
        """Return the callable implementation for the selected method.

        The returned function is looked up on the `method_class` using the
        selected `method_name`.
        """
        return getattr(self.method_class, self.method_name)

    @property
    def method_name(self):
        return self._method_name
    @method_name.setter
    def method_name(self, value):
        """Set method name (validated/converted via method_enum)."""
        self._method_name = self.method_enum(value)

    @property
    def critical_porosity(self):
        """Critical porosity threshold used by some rock models."""
        return self._critical_porosity
    @critical_porosity.setter
    def critical_porosity(self, value):
        """Set critical porosity (numeric)."""
        self._critical_porosity = value 

class MixingModel(EffectiveMediumModel):
    """Wrapper for mixing (VRH or HSW) style effective-medium methods.

    Parameters
    ----------
    method_name : BoundMethods.MixingMethodName
        Enum member selecting the mixing rule.
    upper_weight : float, optional
        Weight used in some mixing recipes (default 0.5).
    """
    def __init__(self, method_name: BoundMethods.MixingMethodName, upper_weight = 0.5):
        super().__init__(BoundMethods, BoundMethods.MixingMethodName, method_name)
        self.upper_weight = upper_weight  

class RockBoundModel(EffectiveMediumModel):
    """Model to calculate the rock moduli using bound models (Voigt-Reuss-Hill or Hashin-Shtrikman).

    Parameters
    ----------
    method_name : BoundMethods.RockMethodName
        Enum member selecting the rock bound method.
    upper_weight : float, optional
        Interpolation weight of upper bound (default 0.5).
    critical_porosity : float, optional
        Porosity value used in modified model.
    """
    def __init__(self, method_name: BoundMethods.RockMethodName, upper_weight = 0.5, critical_porosity = 1):
        super().__init__(BoundMethods, BoundMethods.RockMethodName, method_name)
        self.upper_weight = upper_weight
        self.critical_porosity = critical_porosity

class InclusionModel(EffectiveMediumModel):
    """Model for inclusion-based effective-medium calculations.

    The InclusionModel wraps inclusion/crack style approaches and provides a
    helper to unpack an inclusion set (mapping Inclusion -> fraction).

    Parameters
    ----------
    method_name : InclusionMethods.InclusionMethodName | InclusionMethods.CrackMethodName
        Enum member selecting the inclusion method.
    """
    def __init__(self, method_name: InclusionMethods.InclusionMethodName | InclusionMethods.CrackMethodName):
        super().__init__(InclusionMethods, InclusionMethods.InclusionMethodName, method_name)
               
    @staticmethod
    def unpack_properties(inclusion_set: Dict[Inclusion, np.ndarray]):
        """Extract arrays of inclusion properties from the given inclusion_set.

        Parameters
        ----------
        inclusion_set : dict
            Mapping from Inclusion objects to numeric fractions (np.ndarray).

        Returns
        -------
        tuple of np.ndarray
            (bulks, shears, fractions, shape_types, aspects), where each entry is
            an ndarray aligned with the insertion order of the inclusion_set.
        """
        bulks =       np.array([inclusion.bulk          for inclusion in inclusion_set.keys()])
        shears =      np.array([inclusion.shear         for inclusion in inclusion_set.keys()])
        fractions =   np.array([*inclusion_set.values()])
        shape_types = np.array([inclusion.shape_name    for inclusion in inclusion_set.keys()])
        aspects =     np.array([inclusion.aspect_ratio  for inclusion in inclusion_set.keys()])

        return bulks, shears, fractions, shape_types, aspects

class ContactModel(EffectiveMediumModel):
    """Wrapper for contact-based effective-medium (grain contact) methods.

    This class stores any additional parameters passed via kwargs as attributes
    so that the `method` property can construct a callable that binds those
    additional parameters into the underlying implementation.

    Parameters
    ----------
    method_name : ContactMethods.ContactMethodName
        Enum member selecting the contact algorithm.
    **kwargs :
        Extra parameters required by specific contact algorithms (e.g.
        pressure, adhesion_coef, uncemented_porosity, cement).
    """
    def __init__(self, method_name: ContactMethods.ContactMethodName, **kwargs):
        super().__init__(ContactMethods, ContactMethods.ContactMethodName, method_name)
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    @property
    def method(self) -> callable:
        """Return a callable with required contact parameters bound.

        Different contact algorithms require different additional parameters.
        This property inspects the selected `method_name` and returns a lambda
        that calls the raw implementation with the correct extra arguments.
        """
        raw_method = super().method
        match self.method_name:
            case ContactMethods.ContactMethodName.HertzMindlin:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.pressure, self.adhesion_coef)
            case ContactMethods.ContactMethodName.SoftSand | ContactMethods.ContactMethodName.StiffSand:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.pressure, self.uncemented_porosity, self.adhesion_coef)
            case ContactMethods.ContactMethodName.IntermediateSand | ContactMethods.ContactMethodName.IntermediateStiffSand:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.pressure, self.uncemented_porosity, self.adhesion_coef, self.contact_cement_saturation)
            case ContactMethods.ContactMethodName.ContactCementedSand | ContactMethods.ContactMethodName.SurfaceCementedSand:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.uncemented_porosity, self.cement.bulk, self.cement.shear)
            case ContactMethods.ContactMethodName.IntermediateCementedSand:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.uncemented_porosity, self.cement.bulk, self.cement.shear, self.contact_cement_saturation)
            case ContactMethods.ContactMethodName.Digby | ContactMethods.ContactMethodName.Jenkins | ContactMethods.ContactMethodName.WaltonHydrostaticSmooth | ContactMethods.ContactMethodName.WaltonHydrostaticRough:
                return lambda grain_bulk, grain_shear, porosity, contact_no : raw_method(grain_bulk, grain_shear, porosity, contact_no, self.pressure)
  


# ===================================================================================
# General Trend Models
class TrendType(StrEnum):
    CleanSandstone_Pwave = "Pwave_Pwave_Clean_Sandstone"
    Linear = "Linear"
    Parabolic = "Parabolic"
    PowerLaw = "PowerLaw"
    SquaredParabolic = "SquaredParabolic"
    MultiLinear = "MultiLinear"

    def __str__(self):
        return self.name

class TrendModel:
    """Base class for empirical regression/trend models.

    Stores regression coefficients, regression type and plotting templates for
    x and y. Subclasses implement fit_coefs, forward and inverse as appropriate.

    Attributes
    ----------
    coefs : array-like
        Regression coefficients for the model. Interpretation depends on subclass.
    regression_type : TrendType
        Type of regression implemented by the instance.
    x_template : PropertyTemplate
        Template describing the independent variable (range, units, plotting).
    y_template : PropertyTemplate
        Template describing the dependent variable.
    color : tuple
        RGB color used for plotting the trend line.
    """
    def __init__(self, coefs: tuple = None, regression_type: TrendType = None, x_template: PropertyTemplate = PropertyTemplates().General, y_template:PropertyTemplate = PropertyTemplates().General):
        self.regression_type = regression_type
        self.x_template = x_template
        self.y_template = y_template
        self.coefs = coefs
        self.color = Color()

    @property
    def coefs(self):
        return self._coefs
    @coefs.setter
    def coefs(self, value):
        self._coefs = value

    def fit_coefs(self, x, y):
        """Fit coefficients from data (to be implemented by subclasses)."""
        pass
    
    def inverse(self, y, *args):
        """Return inverse mapping x = f^{-1}(y). Subclasses override as needed."""
        pass
    
    def forward(self, x):
        """Return predicted y for input x. Subclasses override as needed."""
        pass

    def trend_line(self, Nsample = 50):
        """Generate x,y arrays for plotting the model over x_template.plot_range."""
        x = np.linspace(*self.x_template.plot_range, Nsample)
        y = self.forward(x)
        return x,y        

    def draw(self, ax, color = 'r'):
        x, y = self.trend_line()
        ax.plot(x,y, color)

    def __str__(self):
        """Return a compact textual representation of the fitted equation."""
        match self.regression_type:
            case TrendType.Linear:
                return 'Y = {:^+.3f}*X {:^+.3f}'.format(self.coefs[0], self.coefs[1])
            
            case TrendType.Parabolic:
                return 'Y = {:^+.3f}*X**2 {:^+.3f}*X + {:^+.3f}'.format(self.coefs[0], self.coefs[1], self.coefs[2])
            
            case TrendType.SquaredParabolic:
                return 'Y**2 = {:^+.3f}*X**4 {:^+.3f}*X**2 {:^+.3f}'.format(self.coefs[0], self.coefs[1], self.coefs[2])
            
            case TrendType.PowerLaw:
                return 'Y = {:^+.3f} * X**({:^+.3f})'.format(self.coefs[0], self.coefs[1])
            
            case TrendType.MultiLinear:
                text = 'Y = '
                for term in range(np.size(self.x)):
                    text += '{:^+.3}*X{term} '.format(self.coefs[term])
                return text
    
    def __hash__(self):
        """Hash based on an identifying attribute (id expected elsewhere)."""
        return hash(self.id)

class LinearModel(TrendModel):
    """Simple linear regression model y = a*x + b."""
    def __init__(self, coefs: tuple = None, x_template: PropertyTemplate = PropertyTemplates().General, y_template:PropertyTemplate = PropertyTemplates().General):
        super().__init__(coefs, TrendType.Linear, x_template, y_template)

    def fit_coefs(self, x, y):
        """Fit linear coefficients using numpy.polyfit (degree 1)."""
        idx = np.isfinite(x) & np.isfinite(y)
        self.coefs = np.polyfit(x[idx], y[idx], deg=1)
        return self

    def forward(self, x):
        """Evaluate linear model."""
        return self.coefs[0]*x + self.coefs[1]
                
    def inverse(self, y, *args):
        """Inverse mapping x from y for linear model: x = (y - b)/a."""
        return 1/self.coefs[1] * y - self.coefs[2]/self.coefs[1]
            
    def __str__(self) -> str:
        """Return the compact textual representation of the fitted equation."""
        return 'Y = {:^+.2e}*X {:^+.2e}'.format(self.coefs[0], self.coefs[1])

class ParabolicModel(TrendModel):
    """Parabolic model y = ax^2 + bx + c."""
    def __init__(self, coefs: tuple = None, x_template: PropertyTemplate = PropertyTemplates().General, y_template:PropertyTemplate = PropertyTemplates().General):
        super().__init__(coefs, TrendType.Parabolic, x_template, y_template)


    def fit_coefs(self, x, y, order=1):
        """Fit quadratic coefficients using numpy.polyfit (degree 2)."""
        idx = np.isfinite(x) & np.isfinite(y)
        coefs = np.polyfit(x[idx], y[idx], order)
        if order == 1:
            self.coefs = np.array([0, coefs[0], coefs[1]])
        else:
            self.coefs = coefs
        return self

    def forward(self, x):
        """Evaluate quadratic function."""
        return self.coefs[0]*x**2 + self.coefs[1]*x + self.coefs[2]
                
    def inverse(self, y, *args):
        """Return positive root of quadratic for given y (may be multi-valued)."""
        if self.coefs[0] == 0:
            return (y - self.coefs[2]) / self.coefs[1]
        else:
            return -self.coefs[1]/2/self.coefs[0] + np.sqrt(4*self.coefs[0]*(y - (self.coefs[2] - (self.coefs[1]**2)/4/self.coefs[0]) ))/2/self.coefs[0]            

    def __str__(self):
        """Return the compact textual representation of the fitted equation."""
        if self.coefs[0] == 0:
            return 'Y = {:^+.2e} * X {:^+.2e}'.format(self.coefs[1], self.coefs[2])
        else:
            return 'Y = {:^+.2e} * X**2 {:^+.2e} * X {:^+.2e}'.format(self.coefs[0], self.coefs[1], self.coefs[2])

class PowerLawModel(TrendModel):
    """Power-law model y = a * x^b. Coefficients stored as [a, b]."""
    def __init__(self, coefs: tuple = None, x_template: PropertyTemplate = PropertyTemplates().General, y_template:PropertyTemplate = PropertyTemplates().General):
        super().__init__(coefs, TrendType.PowerLaw, x_template, y_template)


    def fit_coefs(self, x, y):
        """Fit power-law by linearizing logs: ln(y) = b*ln(x) + ln(a)."""
        idx = np.isfinite(x) & np.isfinite(y)
        b, loga = np.polyfit(np.log(x[idx]), np.log(y[idx]), deg=1)
        self.coefs = np.zeros(2,)
        self.coefs[0] = np.exp(loga)
        self.coefs[1] = b
        return self

    def forward(self, x):
        """Evaluate power-law."""
        return self.coefs[0]*x**self.coefs[1]
            
    def inverse(self, y, *args):
        """Return x for given y: x = (y/a)^(1/b)."""
        return self.coefs[0]**(-1/self.coefs[1]) * y**(1/self.coefs[1])            

    def __str__(self):
        """Return the compact textual representation of the fitted equation."""
        return 'Y = {:^+.3f} * X**({:^+.3f})'.format(self.coefs[0], self.coefs[1] )

class SquaredParabolicModel(TrendModel):
    """Model of the form y = sqrt(a*x^4 + b*x^2 + c). Useful for some empirical rules."""
    def __init__(self, coefs: tuple = None, x_template: PropertyTemplate = PropertyTemplates().General, y_template:PropertyTemplate = PropertyTemplates().General):
        super().__init__(coefs, TrendType.SquaredParabolic, x_template, y_template)


    def fit_coefs(self, x, y):
        """Fit coefficients by regressing y^2 on x^2 (degree 2)."""
        idx = np.isfinite(x) & np.isfinite(y)
        self.coefs = np.polyfit(x[idx]**2, y[idx]**2, deg=2)
        return self

    def forward(self, x):
        """Evaluate sqrt-quadratic function."""
        return np.sqrt(self.coefs[0]*x**4 + self.coefs[1]*x**2 + self.coefs[2])
                
    def inverse(self, y, *args):
        """Return positive root for x given y. Handles degenerate a==0 case."""
        if self.coefs[0] == 0:
            return np.sqrt(1/self.coefs[1] * y**2 -self.coefs[2]/self.coefs[1])
        else:
            return np.sqrt(-self.coefs[1]/2/self.coefs[0] + np.sqrt(4*self.coefs[0]*(y**2 - (self.coefs[2] - (self.coefs[1]**2)/4/self.coefs[0]) ))/2/self.coefs[0] )

    def __str__(self):
        """Return the compact textual representation of the fitted equation."""
        return 'Y**2 = {:^+.2e} * X**4 {:^+.2e} * X**2 {:^+.2e}'.format(self.coefs[0], self.coefs[1], self.coefs[2])

class MultiLinearModel(TrendModel):
    """Multiple linear regression model: y = c0 + c1*x1 + c2*x2 + ...

    For fitting, X should be provided as a 2D array-like where each column is a
    predictor. The first coefficient c0 is treated as the intercept.
    """
    def __init__(self, coefs: tuple = None, x_template_list: list[PropertyTemplate] = None, y_template:PropertyTemplate =PropertyTemplates().General):
        super().__init__(coefs, TrendType.MultiLinear, x_template_list, y_template)


    def fit_coefs(self, X, y):
        """Fit coefficients using least-squares (scipy.linalg.lstsq).

        X may be 1-D or 2-D; the implementation stacks a leading ones column to
        include the intercept in the solution.
        """
        # idx = np.isfinite(x) & np.isfinite(y)

        A = np.stack((np.ones(np.shape(y)), X), axis=1) #intercept is first term
        self.coefs,_,_,_ = scipy.linalg.lstsq(A, y)
        return self

    def forward(self, X):
        """Evaluate the multilinear model for given predictor vector(s) X."""
        y = self.coefs[0]
        for idx in range(1,np.size(self.coefs)):
            y += self.coefs[idx]*X[idx-1]
        return y
                        
    def inverse(self, y, *args):
        """Inverse not implemented for multilinear models (requires additional constraints)."""
        pass

    def trend_line(self, Nsample = 50):
        """Multi-dimensional trend_line not implemented (requires projection)."""
        pass 
                
    def __str__(self):
        """Return the compact textual representation of the fitted equation."""
        text = 'Y = {:^+.3e} '.format(self.coefs[0])
        for term in range(1,np.size(self.coefs)):
            text += '{:^+.3e} * X{term} '.format(self.coefs[term])
        return text
#----------------------------------------------------------------------------------------------
# Specific Empirical Models
class EmpiricalModels:
    class GardnerModel(PowerLawModel):
        """Gardner density-velocity power-law preset."""
        def __init__(self, trend_name: str | GardnerTrendNames, coefs: tuple = None, template_table = PropertyTemplates()):
            super().__init__(coefs, template_table.PVelocity, template_table.Density)
            self.trend_name = trend_name

    class CastagnaModel(ParabolicModel):
        """Castagna Vp-Vs empirical parabolic relation wrapper."""
        def __init__(self, trend_name: str | CastagnaTrendNames, coefs: tuple = None, template_table = PropertyTemplates()):
            super().__init__(coefs, template_table.PVelocity, template_table.SVelocity)
            self.trend_name = trend_name

    class VernikModel(SquaredParabolicModel):
        """Vernik Vp-Vs squared-parabolic empirical relation wrapper."""
        def __init__(self, trend_name: str | VernikTrendNames, coefs: tuple = None, template_table = PropertyTemplates()):
            super().__init__(coefs, template_table.PVelocity, template_table.SVelocity)
            self.trend_name = trend_name

    class HanModel(MultiLinearModel):
        """Han multi-linear empirical relation (porosity & clay fraction predictors).

        HanModel stores the fitted coefficients and provides helper methods to
        return coefficient sets reorganized for predicting velocity, porosity or
        clay content.
        """
        def __init__(self, trend_name: str | HanTrendNames, coefs: tuple = None, template_table = PropertyTemplates()):
            super().__init__(coefs, [template_table.Porosity, template_table.VolumeFraction])
            self.trend_name = trend_name

        def coefs_for_velocity(self):
            """Return coefficients oriented for predicting velocity (as stored)."""
            return self.coefs

        def coefs_for_porosity(self):
            """Return a coefficient triple to predict porosity from velocity & clay.

            Returns (intercept, clay_coef, velocity_coef) normalized appropriately.
            """
            return np.array(-self.coefs[0], -self.coefs[2], 1) / self.coefs[1] #(intercept, clay_coef, velocity_coef)

        def coefs_for_clay(self):
            """Return coefficients oriented for predicting clay fraction."""
            return np.array(-self.coefs[0], -self.coefs[1], 1) / self.coefs[2] #(intercept, clay_coef, velocity_coef)


if __name__ == "__main__":
    mthd = "vrh"
    method_enum = BoundMethods.MixingMethodName
    print(method_enum(mthd))
    print(method_enum.__qualname__)
