"""
Rock physics constants, containers and empirical coefficients

This module serves as the central repository for all constants, property definitions,
empirical coefficients, and lookup tables used throughout the rokpy package.
It provides standardized definitions for physical properties, material types, and
coefficients of empirical relations used in rock physics modeling and analysis.
"""

from dataclasses import dataclass,  field, fields
from typing import Tuple
import warnings
import numpy as np
from enum import Enum
from rokpy.conversions import api_to_rho, modulus_to_pvelocity, modulus_to_svelocity, velocity_to_bulk
import random
from rokpy.fluid_properties import BatzleWang
    
#Enums=========================================================================================
class PropertyNames(str, Enum):
    General        = "General"
    Vp             = "Vp"
    Vs             = "Vs"
    GR             = "GR"
    Rho             = "Rho"
    Por             = "Por"
    Volume         = "Volume"
    Sat             = "Sat"
    Cal             = "Cal"
    DT             = "DT"
    DTS             = "DTS"
    Temperature     = "Temperature"
    Resistivity     = "Resistivity"
    Modulus         = "Modulus"
    Permeability = "Permeability"
    Depth         = "Depth"
    Time         = "Time"
    PoissonRatio = "PoissonRatio"
    SaturationSet = "SaturationSet"
    VolumeSet     = "VolumeSet"
    Reflectivity = "Reflectivity"
    Thomsen         = "Thomsen"
    SP             = "SP"
    Epsilon         = "Epsilon"
    Delta         = "Delta"
    Gamma         = "Gamma"
    Pressure     = "Pressure"
    CNL             = "CNL"
    Facies         = "Facies"
    Lithology     = "Lithology"
    AI             = "AI"
    GI             = "GI"
    SI             = "SI"
    Pmodulus     = "Pmodulus"
    Shear         = "Shear"
    Bulk         = "Bulk"
    Lambda         = "Lambda"
    Young         = "Young"
    Angle         = "Angle"
    Ratio         = "Ratio"
    VpVs         = "VpVs"
    Gradient     = "Gradient"
    Intercept     = "Intercept"
    Frequency     = "Frequency"
    Viscosity     = "Viscosity"
    Seismic       = "Seismic"


    def __str__(self):
        return self.name

class MineralType(str, Enum):
    General = "General"
    Anhydrite = "Anhydrite"
    Aragonite = "Aragonite"
    Biotite = "Biotite"
    BituminCoal = "Bituminous Coal"
    CaFeldspar = "Ca-Feldspar"
    Calcite = "Calcite"
    Chert = "Chert"
    Chlorite = "Chlorite"
    Dolomite = "Dolomite"
    DryClay = "Dry Clay"
    Glauconite = "Glauconite"
    Gypsum = "Gypsum"
    Halite = "Halite"
    Hornblende = "Hornblende"
    Illite = "Illite"
    Kaolinite = "Kaolinite"
    Kerogen = "Kerogen"
    KFeldspar = "K-Feldspar"
    Magnesite = "Magnesite"
    Magnetite = "Magnetite"
    Muscovite = "Muscovite"
    NaFeldspar = "Na-Feldspar"
    Olivine = "Olivine"
    Orthoclase = "Orthoclase"
    Plagioclase = "Plagioclase"
    Pyrite = "Pyrite"
    Pyroxene = "Pyroxene"
    Quartz = "Quartz"
    Shale = "Shale"
    Siderite = "Siderite"
    Smectite = "Smectite"

    def __str__(self):
        return self.name

class FluidType(str, Enum):
    General = "General"
    Dry = "Dry"
    Brine = "Brine"
    Oil = "Oil"
    Gas = "Gas"
    Condensate = "Condensate"
    CO2 = "CO2"
    HeavyOil = "HeavyOil"
    Steam = "Steam"

    def __str__(self):
        return self.name

#Containers===================================================================================
@dataclass
class ElasticPropertySet:
    """A container for elastic properties of arbitrary materials such as minerals and fluids.

    parameters
    ----------
    density : float
        Density in g/cc
    bulk : float
        Bulk modulus in GPa
    shear : float
        Shear modulus in GPa
    type : str, optional
        The type of the mineral or rock (default is "Mineral")
    color : tuple, optional
    """
    density: float = field(metadata = {'unit': 'g/cc'})
    bulk   : float = field(metadata = {'unit': 'GPa'})
    shear  : float = field(metadata = {'unit': 'Gpa'})
    type   : str   = field(default = "Mineral")
    color  : tuple = field(default_factory=lambda: Color())
    
    def __post_init__(self):
        self.id = random.getrandbits(128)

    @property
    def p_velocity(self):
        return modulus_to_pvelocity(self.bulk, self.shear, self.density)
    
    @property
    def s_velocity(self):
        return modulus_to_svelocity(self.shear, self.density)

    @property
    def abbreviation(self):
        return self.type[:4]

    def set_properties(self, bulk, shear, density):
        self.bulk = bulk    
        self.shear = shear
        self.density = density
        
    def __eq__( self, other: object) -> bool:
        return  self.type    == other.type and \
                self.density == other.density and \
                self.bulk    == other.bulk and \
                self.shear   == other.shear
    
    def __str__(self):
        
        return 'type: {:<15s}| rho: {:<5.3f}\t| K: {:<8.3f}| G: {:<8.3f}| Vp: {:<9.2f}| Vs: {:<9.2f}'.format \
                (self.type  , self.density    , self.bulk   , self.shear  , self.p_velocity, self.s_velocity)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class PropertyTemplate:
    """A container for property template specifications.

    parameters
    ----------
    type : str
        The name of the property
    unit : str
        The unit of the property
    plot_range : tuple[float]
        The display range for plotting
    symbol : str
        The symbol (abbreviation) to use for plotting
    is_key_property : bool, optional
        Whether this property is a key property (default is False)
    alias_list : list[str], optional
        A list of alias names for the property (default is empty list)
    color : tuple, optional
        The color to use for plotting (default is random color)
    """
    type: str
    unit: str
    plot_range: tuple[float]
    symbol: str
    is_key_property: bool = field(default=False)
    alias_list: list[str] = field(default_factory=lambda: [])
    color: tuple = field(default_factory=lambda: Color())

    def __str__(self):
        return 'type: {:<15s}\t| unit: {:<8s}\t| Display Range: {:<10s}\t| symbol: {:<7s}\t| is_key: {:<7s}'.format( \
                self.type      , self.unit      , self.plot_range.__str__(), self.symbol     ,self.is_key_property.__str__())

@dataclass
class CastagnaCoefficients:
    """A container for coefficients of Castagna's relation (S-velocity in m/s vs. P-velocity in m/s)
    
    Parameters
    ----------  
    a2 : float
        Coefficients of quadratic P-velocity term
    a1 : float
        Coefficients of linear P-velocity term
    a0 : float
        Constant term
    type : str, optional
        The type of the mineral or rock (default is "Mineral")
    color : tuple, optional
    """
    a2: float
    a1: float
    a0: float
    type: str = field(default="Mineral")
    color: tuple = field(default_factory=lambda: Color())

    @property
    def coefficients(self, ):
        """Return the coefficients as a tuple.
        """
        return (self.a2, self.a1, self.a0)
    @coefficients.setter
    def coefficients(self, values):
        self.a2, self.a1, self.a0 = values

    def __str__(self):
        return 'type: {:<25s}\t| a2: {:<10.7f}\t| a1: {:<8.3f}| a0: {:<8.3f}'.format( \
                self.type      , self.a2       , self.a1     , self.a0)

@dataclass
class GardnerCoefficients:
    """A container for coefficients of Gardner's relation (Density in g/cc vs. P-velocity in m/s)
    
    Parameters
    ----------  
    multiplier : float
        multiplier term
    exponent : float
        exponent term
    type : str, optional
        The type of the mineral or rock (default is "Mineral")
    color : tuple, optional
    """
    multiplier: float
    exponent: float
    type: str = field(default="Mineral")
    color: tuple = field(default_factory=lambda: Color())

    @property
    def coefficients(self):
        """Return the coefficients as a tuple.
        """
        return (self.multiplier, self.exponent)
    @coefficients.setter
    def coefficients(self, values):
        self.multiplier, self.exponent = values

    def __str__(self):
        return 'type: {:<20s}\t| multiplier: {:<5.3f}\t| exponent: {:<8.3f}'.format( \
                self.type      , self.multiplier       , self.exponent )
    
@dataclass
class HanCoefficients:
    """A container for coefficients of Han's relation (Velocity in m/s vs. porosity and clay content)
    
    Parameters
    ----------  
    intercept : float
        intercept term
    phi_coef : float
        porosity coefficient
    clay_coef : float
        clay coefficient
    type : str, optional
        The type of the mineral or rock (default is "Mineral")
    color : tuple, optional
    """
    intercept: float
    phi_coef: float
    clay_coef: float
    type: str = field(default='General')
    color: tuple = field(default_factory=lambda: Color())

    @property
    def coefficients(self):
        """Return the coefficients as a tuple.
        """
        return (self.intercept, self.phi_coef, self.clay_coef)
    @coefficients.setter
    def coefficients(self, values):
        self.intercept, self.phi_coef, self.clay_coef = values

    def __str__(self):
        return 'type: {:<30s}\t| a_0: {:<5.3f}\t| a_phi: {:<8.3f}| a_C: {:<8.3f}'.format( \
                self.type    , self.intercept, self.phi_coef, self.clay_coef)

@dataclass
class VernikCoefficients:
    """A container for coefficients of Vernik's relation (S-velocity in m/s vs. P-velocity in m/s)
    
    Parameters
    ----------  
    a4 : float
        Coefficients of squared quadratic P-velocity term
    a2 : float
        Coefficients of squared P-velocity term
    a0 : float
        Constant term
    type : str, optional
        The type of the mineral or rock (default is "Mineral")
    """    
    a4: float
    a2: float
    a0: float
    type: str = field(default="Mineral")
    color: tuple = field(default_factory=lambda: Color())

    @property
    def coefficients(self, ):
        """Return the coefficients as a tuple.
        """
        return (self.a4, self.a2, self.a0)
    @coefficients.setter
    def coefficients(self, values):
        self.a4, self.a2, self.a0 = values
    
    def __str__(self):
        return 'type: {:<15s}\t| a4: {:<5.3f}\t| a2: {:<8.3f}| a0: {:<8.3f}'.format( \
                self.type      , self.a4       , self.a2     , self.a0)

#==============================================================================================
#==============================================================================================
class VernikTrendNames(str, Enum):
    GeneralLine = "General"
    LimestoneLine = "Limestone"
    DolomiteLine = "Dolomite"
    SandstoneLine = "Sandstone"
    ShaleLine = "Shale"

    def __str__(self):
        return self.name

class GardnerTrendNames(str, Enum):
    GeneralLine = "General"
    ShaleLine = "Shale"
    SandstoneLine = "Sandstone"
    LimestoneLine = "Limestone"
    DolomiteLine = "Dolomite"
    AnhydriteLine = "Anhydrite"

    def __str__(self):
        return self.name

class CastagnaTrendNames(str, Enum):
    GeneralLine = "General"
    LimestoneLine = "Limestone"
    DolomiteLine = "Dolomite"
    SandstoneLine = "Sandstone"
    ShaleLine = "Shale"
    MudrockLine = "Mudrock"
    HanLine = "Han"
    HanLine_LowClay = "HanLow"   #Clay<0.25
    HanLine_HighClay = "HanHigh" #Clay>0.25
    HanLine_LowPorosity = "HanLowPor"   #Por<0.15
    HanLine_HighPorosity = "HanHighPor" #Por>0.15
    CoalLine = "Coal"

    def __str__(self):
        return self.name

class HanTrendNames(str, Enum):
    General = "General"
    CleanSandstone_Pwave = "Pwave_Pwave_Clean_Sandstone"
    DrySandstone_40MPa_Pwave = "Pwave_DrySandstone@40MPa"
    Sandstone_40MPa_Pwave = "Pwave_Sandstone@40MPa"
    Sandstone_30MPa = "Pwave_Sandstone@30MPa"
    Sandstone_20MPa_Pwave = "Pwave_Sandstone@20MPa"
    Sandstone_10MPa_Pwave = "Pwave_Sandstone@10MPa"
    Sandstone_5MPa_Pwave = "PSwave_Sandstone@5MPa"
    FrioSandstone_Pwave = "Pwave_FrioSandstone"
    TosayaNurSandstone_Pwave = "Pwave_TosayaNurSandstone"
    CleanSandstone_Swave = "Swave_Pwave_Clean_Sandstone"
    DrySandstone_40MPa_Swave = "Swave_DrySandstone@40MPa"
    Sandstone_40MPa_Swave = "Swave_Sandstone@40MPa"
    Sandstone_30MPa_Swave = "Swave_Sandstone@30MPa"
    Sandstone_20MPa_Swave = "Swave_Sandstone@20MPa"
    Sandstone_10MPa_Swave = "Swave_Sandstone@10MPa"
    Sandstone_5MPa_Swave = "Swave_Sandstone@5MPa"
    FrioSandstone_Swave = "Swave_FrioSandstone"
    TosayaNurSandstone_Swave = "Swave_TosayaNurSandstone"

    def __str__(self):
        return self.name

class LithoTypes(str, Enum):
    General = "General"
    Shale = "Shale"
    Sandstone = "Sandstone"
    Limestone = "Limestone"
    Dolomite = "Dolomite"
    Anhydrite = "Anhydrite"
    Mudrock = "Mudrock"
    Coal = "Coal"

    def __str__(self):
        return self.name    
#Tables ======================================================================================
#Property Tables------------------------------------------------------------------------------
@dataclass
class DataTable():
    
    user_items   : dict = field(default_factory = lambda: {} )

    @property
    def table_type(self):
        return dict

    def add_item(self, item_name: str, item_value):
        if isinstance(item_value, self.table_type):
            self.user_items[item_name] = item_value
            self.__setattr__(item_name, item_value)
        else:
            raise ValueError(f"item_value must be a {self.table_type.__name__} object")

    def items(self):
        items = self.user_items
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, self.table_type):
                items[field.name] = value
        return items.items()

    def __str__(self):
        text = ''
        for field, value in self.items():
            text += '{:<20s} : {:>s}\n'.format(field, value.__str__())
        return text
    
@dataclass
class PropertyTemplates(DataTable):
    General       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.General       ,"unitless" , ( None,    None),"X"     ,  True, [], Color(0, 0, 0)))
    PVelocity     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Vp            ,"m/s"      , ( 2000,    7000),"Vp"    ,  True, [], Color(0, 0, 0)))
    SVelocity     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Vs            ,"m/s"      , ( 1000,    3500),"Vs"    ,  True, [], Color(0, 0, 0)))
    GR            : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.GR            ,"API"      , (    0,     150),"GR"    ,  True, [], Color(0, 0, 0)))
    Density       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Rho           ,"g/cm3"    , (    2,       3),"Rho"   ,  True, [], Color(0, 0, 0)))
    Porosity      : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Por           ,"fract"    , (    0,     0.4),"Por"   ,  True, [], Color(0, 0, 0)))
    VolumeFraction: PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Volume        ,"fract"    , (    0,       1),"V"     ,  True, [], Color(0, 0, 0)))
    Saturation    : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Sat           ,"fract"    , (    0,       1),"Sat"   ,  True, [], Color(0, 0, 0)))
    Caliper       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Cal           ,"m"        , ( 0.15,     0.5),"Cal"   ,  True, [], Color(0, 0, 0)))
    PSonic        : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.DT            ,"us/ft"    , (  140,      40),"DT"    ,  True, [], Color(0, 0, 0)))
    SSonic        : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.DTS           ,"us/ft"    , (609.6,  121.92),"DTS"   ,  True, [], Color(0, 0, 0)))
    Temperature   : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Temperature   ,"degC"     , (    0,     120),"T"     , False, [], Color(0, 0, 0)))
    Resistivity   : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Resistivity   ,"Ohmm"     , (  0.2,     100),"RT"    , False, [], Color(0, 0, 0)))
    Modulus       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Modulus       ,"GPa"      , (    0,     100),"Mod"   , False, [], Color(0, 0, 0)))
    Permeability  : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Permeability  ,"mD"       , (    0,    4000),"k"     , False, [], Color(0, 0, 0)))
    Depth         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Depth         ,"m"        , (    0,    5000),"D"     , False, [], Color(0, 0, 0)))
    Time          : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Time          ,"ms"       , (    0,    5000),"TWT"   , False, [], Color(0, 0, 0)))
    PoissonRatio  : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.PoissonRatio  ,"unitless" , (    0,     0.5),"nu"    , False, [], Color(0, 0, 0)))
    SaturationSet : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.SaturationSet ,"fract"    , (    0,       1),"SatSet", False, [], Color(0, 0, 0)))
    VolumeSet     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.VolumeSet     ,"fract"    , (    0,       1),"VolSet", False, [], Color(0, 0, 0)))
    Reflectivity  : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Reflectivity  ,"amplitude", (   -1,       1),"R"     , False, [], Color(0, 0, 0)))
    SP            : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.SP            ,"mV"       , (  -60,      60),"SP"    , False, [], Color(0, 0, 0)))
    Epsilon       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Epsilon       ,"unitless" , ( -1.5,     1.5),"ep"    , False, [], Color(0, 0, 0)))
    Delta         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Delta         ,"unitless" , ( -1.5,     1.5),"del"   , False, [], Color(0, 0, 0)))
    Gamma         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Gamma         ,"unitless" , ( -1.5,     1.5),"gam"   , False, [], Color(0, 0, 0)))
    Pressure      : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Pressure      ,"MPa"      , (    0,     100),"P"     , False, [], Color(0, 0, 0)))
    CNL           : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.CNL           ,"fract"    , ( 0.75,   -0.15),"CNL"   , False, [], Color(0, 0, 0)))
    Facies        : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Facies        ,"unitless" , (    0,      10),"F"     , False, [], Color(0, 0, 0)))
    Lithology     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Lithology     ,"unitless" , (    0,      15),"Lith"  , False, [], Color(0, 0, 0)))
    PImpedance    : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.AI            ,"g/cm3_m/s", ( 4000,   12000),"AI"    , False, [], Color(0, 0, 0)))
    SImpedance    : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.GI            ,"g/cm3_m/s", ( 4000,   12000),"SI"    , False, [], Color(0, 0, 0)))
    Pmodulus      : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Pmodulus      ,"GPa"      , (    0,     100),"M"     , False, [], Color(0, 0, 0)))
    Shear         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Shear         ,"GPa"      , (    0,     100),"G"     , False, [], Color(0, 0, 0)))
    Bulk          : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Bulk          ,"GPa"      , (    0,     100),"K"     , False, [], Color(0, 0, 0)))
    Lambda        : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Lambda        ,"GPa"      , (    0,     100),"L"     , False, [], Color(0, 0, 0)))
    Young         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Young         ,"GPa"      , (    0,     100),"E"     , False, [], Color(0, 0, 0)))
    Angle         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Angle         ,"deg"      , (    0,     180),"theta" , False, [], Color(0, 0, 0)))
    Ratio         : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Ratio         ,"unitless" , (    0,       1),"Ratio" , False, [], Color(0, 0, 0)))
    VpVsRatio     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.VpVs          ,"unitless" , (    0,       4),"VpVs"  , False, [], Color(0, 0, 0)))
    Gradient      : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Gradient      ,"unitless" , (   -1,       1),"G"     , False, [], Color(0, 0, 0)))
    Intercept     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Intercept     ,"unitless" , (   -1,       1),"R0"    , False, [], Color(0, 0, 0)))
    Frequency     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Frequency     ,"Hz"       , (    0,     250),"f"     , False, [], Color(0, 0, 0)))
    Viscosity     : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Viscosity     ,"cP"       , (    0, 5000000),"mu"    , False, [], Color(0, 0, 0)))
    Seismic       : PropertyTemplate = field(default_factory = lambda: PropertyTemplate( PropertyNames.Seismic       ,"seis"     , (    -1,      1),"seis"  , False, [], Color(0, 0, 0)))

    @property
    def table_type(self):
        return PropertyTemplate

@dataclass
class MineralsPropertyTable (DataTable):
    General     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1e-16,  0.00,  0.00, MineralType.General    , Color( 99, 99, 99) )  )
    Anhydrite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.00,  66.50, 34.00, MineralType.Anhydrite  , Color( 99, 99, 99) )  )
    Aragonite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.94,  47.00, 39.00, MineralType.Aragonite  , Color(181, 61, 84) )  )
    Biotite     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.00,  50.00, 27.00, MineralType.Biotite    , Color(153,153,153) )  )
    BituminCoal : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1.24,   4.62,  2.61, MineralType.BituminCoal, Color( 61, 61, 61) )  )
    CaFeldspar  : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.81,  75.70, 37.00, MineralType.CaFeldspar , Color(255, 51,204) )  )
    Calcite     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.71,  63.70, 31.70, MineralType.Calcite    , Color(204,204,255) )  )
    Chert       : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.63,  42.20, 37.00, MineralType.Chert      , Color(181,  3, 99) )  )
    Chlorite    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.76, 146.25, 68.15, MineralType.Chlorite   , Color( 74, 89, 56) )  )
    Dolomite    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.87,  94.90, 45.00, MineralType.Dolomite   , Color( 51,  0,255) )  )
    DryClay     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.67,  22.10,  8.50, MineralType.DryClay    , Color(145, 79,  0) )  )
    Glauconite  : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.67,  15.00, 10.00, MineralType.Glauconite , Color(255, 51,204) )  )
    Gypsum      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.31,  58.00, 30.00, MineralType.Gypsum     , Color(  0,255,255) )  )
    Halite      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.16,  25.20, 15.30, MineralType.Halite     , Color(255,153,153) )  )
    Hornblende  : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.12,  87.00, 43.00, MineralType.Hornblende , Color(255,153,153) )  )
    Illite      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.67,  33.50, 15.60, MineralType.Illite     , Color(194, 89, 38) )  )
    Kaolinite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.44,  12.00,  6.00, MineralType.Kaolinite  , Color(237, 89, 94) )  )
    Kerogen     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1.30,   4.20,  3.60, MineralType.Kerogen    , Color(105, 74, 33) )  )
    KFeldspar   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.57,  59.40, 30.30, MineralType.KFeldspar  , Color(255,179,  0) )  )
    Magnesite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.01, 114.00, 68.00, MineralType.Magnesite  , Color( 51, 51, 51) )  )
    Magnetite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 5.18, 160.29, 91.38, MineralType.Magnetite  , Color(145, 79,  0) )  )
    Muscovite   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.81,  57.60, 35.20, MineralType.Muscovite  , Color(204,204,204) )  )
    NaFeldspar  : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.63,  58.50, 35.50, MineralType.NaFeldspar , Color(255,  0,  0) )  )
    Olivine     : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.31, 129.00, 79.00, MineralType.Olivine    , Color(  0,255,255) )  )
    Orthoclase  : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.52,  17.96, 32.41, MineralType.Orthoclase , Color(204,204,204) )  )
    Plagioclase : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.59,  55.80, 33.30, MineralType.Plagioclase, Color(153,153,153) )  )
    Pyrite      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 5.02, 158.00,149.00, MineralType.Pyrite     , Color(102,255,  0) )  )
    Pyroxene    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.32, 104.00, 61.00, MineralType.Pyroxene   , Color( 51,255,153) )  )
    Quartz      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.65,  36.60, 45.00, MineralType.Quartz     , Color(255,255,  0) )  )
    Shale       : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.35,  11.40,  3.00, MineralType.Shale      , Color( 18,140, 26) )  )
    Siderite    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 3.96, 123.70, 51.00, MineralType.Siderite   , Color(219,150, 61) )  )
    Smectite    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 2.54,  24.50,  9.80, MineralType.Smectite   , Color(138,  8, 10) )  )

    @property
    def table_type(self):
        return ElasticPropertySet
    
@dataclass
class FluidsPropertyTable (DataTable):
    Dry        : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1e-16, 0.000, 0.000, FluidType.Dry     , Color(255,255,255) )  )
    General    : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1.00, 2.250, 0.000, FluidType.General  , Color( 99, 99, 99) )  )
    Brine      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1.00, 2.560, 0.000, FluidType.Brine    , Color(153,153,255) )  )
    Oil        : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 0.80, 1.152, 0.000, FluidType.Oil      , Color(153,255,153) )  )
    Gas        : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 0.15, 0.038, 0.000, FluidType.Gas      , Color(255,153,153) )  )
    Condensate : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 0.45, 0.325, 0.000, FluidType.Condensate,Color(255,153,  0) )  )
    CO2        : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 0.15, 0.038, 0.000, FluidType.CO2      , Color(191,191,191) )  )
    HeavyOil   : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 1.10, 2.757, 0.044, FluidType.HeavyOil , Color(181,255,181) )  )
    Steam      : ElasticPropertySet = field(default_factory = lambda: ElasticPropertySet( 0.015,0.004, 0.000, FluidType.Steam    , Color(230,  0,255) )  )

    @property
    def table_type(self):
        return ElasticPropertySet
    
    def calculate_brine (self, T, P, salinity):
        """
        Compute brine density and bulk modulus and set into fluids table

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        salinity : array_like
            Salinity in ppm.

        Returns
        -------
        density : ndarray
            Brine density in g/cm³.
        bulk : ndarray
            Brine bulk modulus in GPa.
        """
        self.Brine.density, self.Brine.bulk  = BatzleWang.brine_properties(T, P, salinity)

    def calculate_oil (self, T, P, api, gor, solutiongas_gravity):
        """
        Compute live oil density and bulk modulus and set into fluids table

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        solutiongas_gravity : array_like
            Specific gravity dissolved gas

        Returns
        -------
        density : ndarray
            Oil density in g/cm³.
        bulk : ndarray
            Oil bulk modulus in GPa.
        """
        self.Oil.density, self.Oil.bulk  = BatzleWang.oil_properties(T, P, api, gor, solutiongas_gravity)

    def calculate_gas (self, T, P, freegas_gravity):
        """
        Compute natural gas density and bulk modulus and set into fluids table

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        freegas_gravity : array_like
            Free-Gas specific gravity.

        Returns
        -------
        density : ndarray
            Gas density in g/cm³.
        bulk : ndarray
            Gas bulk modulus in GPa.
        """
        self.Gas.density, self.Gas.bulk  = BatzleWang.gas_properties(T, P, freegas_gravity)

    def calculate_co2 (self, T, P, freegas_gravity):
        """
        Compute CO₂ density and bulk modulus using Xu (2006) modification.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        freegas_gravity : array_like
            CO₂ specific gravity.

        Returns
        -------
        density : ndarray
            CO₂ density in g/cm³.
        bulk : ndarray
            CO₂ bulk modulus in GPa.
        """       
        self.CO2.density, self.CO2.bulk  = BatzleWang.co2_properties(T, P, freegas_gravity)

    def calculate (self, T, P, salinity, api, gor, solutiongas_gravity, freegas_gravity):
        """
        Compute Brine, oil, gas properties and set into fluids table

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        salinity : array_like
            Salinity in ppm.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        solutiongas_gravity : array_like
            Specific gravity dissolved gas
        freegas_gravity : array_like
            Free-Gas/Co2 specific gravity.

        Returns
        -------
        density : ndarray
            Oil density in g/cm³.
        bulk : ndarray
            Oil bulk modulus in GPa.
        """
        self.calculate_brine (T, P, salinity)
        self.calculate_oil (T, P, api, gor, solutiongas_gravity)
        self.calculate_gas (T, P, freegas_gravity)
        self.calculate_co2 (T, P, freegas_gravity)
 
#Empirical Tables------------------------------------------------------------------------------
@dataclass
class GardnerTable (DataTable):
    GeneralLine   : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.309, 0.250, LithoTypes.General  , Color(100,100,100))   )
    ShaleLine     : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.281, 0.265, LithoTypes.Shale    , Color( 17,139, 25))   )
    SandstoneLine : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.274, 0.261, LithoTypes.Sandstone, Color(255,255,  0))   )
    LimestoneLine : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.095, 1.386, LithoTypes.Limestone, Color(204,204,255))   )
    DolomiteLine  : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.305, 0.252, LithoTypes.Dolomite , Color( 52,  0,255))   )
    AnhydriteLine : GardnerCoefficients = field(default_factory = lambda: GardnerCoefficients( 0.725, 0.160, LithoTypes.Anhydrite, Color( 52,  0,255))   )

    @property
    def table_type(self):
        return GardnerCoefficients

@dataclass
class CastagnaTable (DataTable):
    GeneralLine          : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.500,  0.0,   LithoTypes.General  , Color(100,100,100))   )
    LimestoneLine        : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients(-0.000055,  1.017, -1031., LithoTypes.Limestone, Color(204,204,255))   )
    DolomiteLine         : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.583, -78.,   LithoTypes.Dolomite , Color( 52,  0,255))   )
    SandstoneLine        : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.804, -856.,  LithoTypes.Sandstone, Color(255,255,  0))   )
    ShaleLine            : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.770, -867.,  LithoTypes.Shale    , Color( 17,139, 25))   )
    MudrockLine          : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.862, -1172., LithoTypes.Mudrock  , Color(136,197, 13))   )
    HanLine              : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.794, -787.,  LithoTypes.Sandstone, Color(220,220,  0))   )
    HanLine_LowClay      : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.754, -657.,  LithoTypes.Sandstone, Color( 96,178, 17))   )
    HanLine_HighClay     : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.842, -1099., LithoTypes.Sandstone, Color(176,216,  8))   )
    HanLine_LowPorosity  : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.853, -1137., LithoTypes.Sandstone, Color(240,240,  0))   )
    HanLine_HighPorosity : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.756, -662.,  LithoTypes.Sandstone, Color(200,200,  0))   )
    HanLine              : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.794, -787.,  LithoTypes.Sandstone, Color(220,220,  0))   )
    CoalLine             : CastagnaCoefficients = field(default_factory = lambda: CastagnaCoefficients( 0.0,       0.481,  3.8,   LithoTypes.Coal     , Color( 60, 60, 60))   )

    @property
    def table_type(self):
        return CastagnaCoefficients

@dataclass
class HanTable (DataTable):
    CleanSandStone_PWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 6080, -8060,  0.00, LithoTypes.Sandstone, Color(204,204,255))   )
    CleanSandStone_SWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 4060, -6280,  0.00, LithoTypes.Sandstone, Color( 52,  0,255))   )
    DrySandStone_PWave       : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5041, -6350, -2870, LithoTypes.Sandstone, Color(255,255,  0))   )
    DrySandStone_SWave       : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3570, -4570, -1830, LithoTypes.Sandstone, Color( 17,139, 25))   )
    SandStone_40MP_PWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5590, -6930, -2180, LithoTypes.Sandstone, Color(136,197, 13))   )
    SandStone_40MP_SWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3520, -4910, -1890, LithoTypes.Sandstone, Color(220,220,  0))   )
    SandStone_30MP_PWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5550, -6960, -2180, LithoTypes.Sandstone, Color( 96,178, 17))   )
    SandStone_30MP_SWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3470, -4840, -1870, LithoTypes.Sandstone, Color(176,216,  8))   )
    SandStone_20MP_PWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5490, -6940, -2170, LithoTypes.Sandstone, Color(240,240,  0))   )
    SandStone_20MP_SWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3390, -4730, -1810, LithoTypes.Sandstone, Color(200,200,  0))   )
    SandStone_10MP_PWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5390, -7080, -2130, LithoTypes.Sandstone, Color(240,240,  0))   )
    SandStone_10MP_SWave     : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3290, -4730, -1740, LithoTypes.Sandstone, Color(200,200,  0))   )
    SandStone_5MP_PWave      : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5260, -7080, -2020, LithoTypes.Sandstone, Color(240,240,  0))   )
    SandStone_5MP_SWave      : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3160, -4770, -1640, LithoTypes.Sandstone, Color(200,200,  0))   )
    FrioSandstone_PWave      : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5810, -9420, -2210, LithoTypes.Sandstone, Color(250,250,  0))   )
    FrioSandstone_SWave      : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3890, -7070, -2040, LithoTypes.Sandstone, Color(250,250,  0))   )
    TosayaNurSandstone_PWave : HanCoefficients = field(default_factory = lambda: HanCoefficients( 5800, -8600, -2400, LithoTypes.Sandstone, Color(250,250,  0))   )
    TosayaNurSandstone_SWave : HanCoefficients = field(default_factory = lambda: HanCoefficients( 3700, -6300, -2100, LithoTypes.Sandstone, Color(250,250,  0))   )

    @property
    def table_type(self):
        return HanCoefficients

@dataclass
class VernikTable (DataTable):
    SandstoneLine : VernikCoefficients = field(default_factory = lambda: VernikCoefficients( 2.84e-9, 3.72e-1, -1.267e6,  LithoTypes.Sandstone       , Color(255,255,  0))   )
    ShaleLine     : VernikCoefficients = field(default_factory = lambda: VernikCoefficients( 2.84e-9, 2.87e-1, -0.790e6,  LithoTypes.Shale           , Color( 17,139, 25))   )

    @property
    def table_type(self):
        return VernikCoefficients



class Color(list):
    """
    A simple RGB color class with tuple behavior.
    """

    def __init__(self, r: int =None, g: int =None, b: int =None):
        super().__init__([r, g, b])
        if r is None:
            r = np.random.randint(0, 256)
        if g is None:
            g = np.random.randint(0, 256)
        if b is None:
            b = np.random.randint(0, 256)
        self[0]  = r
        self[1]  = g
        self[2]  = b

    @property
    def r(self):
        """Return the red component of the color."""
        return self[0]  
    @property
    def g(self):
        """Return the green component of the color."""
        return self[1]  
    @property
    def b(self):
        """Return the blue component of the color."""
        return self[2]  
    
    @property
    def hex(self):
        """Return the hexadecimal representation of the color."""
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    @property
    def decimal(self):
        """Return the decimal representation of the color."""
        return (self.r / 255, self.g / 255, self.b / 255)
    
    def random_color(self):
        """Generate a random color."""
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        return Color(r, g, b)



if __name__ == "__main__":
    # Usage
    c = Color()  # Red
    print(c)          # Color(r=255, g=0, b=0)
    print(c.hex)      # #ff0000
    print(c[0])       # 255 (tuple behavior works)
    print(c.r)        # 255
    print(c.decimal)

    # Get random color (returns new instance)
    c_random = c.random_color()
    print(c_random)   # Random color
