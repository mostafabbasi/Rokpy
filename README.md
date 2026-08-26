![A ObjectOrient Python Package for Rock Physics Modeling](rokpy.png)

# What is rokpy?

`rokpy` is a comprehensive object-oriented Python package for building and tuning rock-physics models.

**by: Mostafa Abbasi**


# Installation

To install the package, simply use the following pip command in your terminal:

```bash
pip install rokpy
```
# Documentaion

For a detailed documentation of the package see:

[https://rokpy.readthedocs.io/en/latest/](https://rokpy.readthedocs.io/en/latest/)

# Tutorials

The package has an `examples` directory containing some tutorials notebooks. This tutorials give you a good primary outlook to implement different models and codes. These notebooks also show you different capabilities of the codes beyond just modeling.

Don't limit yourself to these notebooks and also dive into the documentations to explore different classes, attributes and functions.

You may also find these tutorials at:

[https://rokpy.readthedocs.io/en/latest/tutorials/](https://rokpy.readthedocs.io/en/latest/tutorials/)




# Introduction

`rokpy` is a comprehensive python package for Rock-physical modeling that includes a wide variety of models and relations. the package follows 
an object-oriented programming (OOP) scheme for modeling the rock and its components. According to this scheme, a rock is object with many 
different attributes such as porosity, minerals, fluids, inclusions, etc. and also many different properties such as velocities, bulk and shear 
moduli, density, etc. All these properties are already included into a rock (or any of its components), therefore, It doesn't require the user 
to write code for any of these features.

Once a rock model is build and it's hyper parameters such as pore shapes or fluid and mineral properties are fine-tuned for a given formation in 
a filed, one may use the model for another field. Therefore, using the `rokpy`, you may **"Model here, apply there"**.

In addition to pre-designed models of rock and its components, these users may also use an extensive library of rock-physical relations and methods
to build their own models from the scratch. These methods are categorized into different modules such as **bound methods**, **inclusion methods**, **contact methods** and **fluid effect methods**.

The package methods are mainly written on the basis of the formulations and descriptions given in *The Rock-Physics handbook (Mavko et al., 2020)*,
however, the methods are not limited to this book and many other key literatures are also reviewed to build the package. It worth noting that I 
have been tried to verify the codes by regenerating the plots in the book and and corresponding papers.

# Highlights

- Rock-physics modeling tools implemented in Python
- Object-Oriented Scheme of programming
- Detailed documentation and tutorials
- A widly comprehensive library of rock-physics models
- Forward AVO modeling tools
- NumPy-based scientific data handling

# Package Organization

`rokpy` package is organized into following modules

| Module | Description |
| --- | --- |
| `effective_medium` | Library of different effective medium relations |
| `materials` | Rich classes of different materials such as Rocks, Minerals and Fluids |
| `models` | Library wrapping theoritical and empirical relations into easy to use models |
| `AVO` | Library of forward AVO relations |
| `fluid_properties` | Library of relations to calculate in-situ fluid properties |
| `backus` | Library of relations to estimate the effective average (Backus) properties |
| `constants` | Reference tables of standard constans and empirical coefficients |
| `utilities` | Set of auxilliary tools and relations used throughout the package |
| `conversions` | Library of functions to convert different elastic properties |
| `visualization` | Well-log plotting tools |

# An Example

Assuming that we have already imported well data in our workspace, following is an example of how we can implement a Xu-White model by `rokpy`:

```python
from rokpy.materials import Mineral, Fluid, MineralSet, FluidSet, InclusionRock
from rokpy.constants import MineralsPropertyTable, FluidsPropertyTable
from rokpy.effective_medium import Inclusion
from rokpy.conversions import psi_to_mpa

# Build a table of default mineral properties
minerals = MineralsPropertyTable()

# Build mineral objects by corresponding properties
clay = Mineral(minerals.DryClay)
quartz = Mineral(minerals.Quartz)
calcite = Mineral(minerals.Calcite)

# Setup the mineral set (rock matrix)
mineralset = MineralSet({clay    : vclay,
                         calcite : vcal ,
                         quartz  : vqz  },
                         mixing_method = 'voigt_reuss_hill')
#----------------------
# Build a table of default fluid properties
fluids = FluidsPropertyTable()

# Modify fluid properties for in-situ conditions
fluids.calculate_brine(T=40, P=psi_to_mpa(2984.), salinity=50000)

# Build fluid objects by corresponding properties
brine = Fluid(fluids.Brine)
oil = Fluid(fluids.Oil)

# Setup the fluid set
fluidset = FluidSet({brine : swt,
                     oil   : 1-swt}

# Define the rock
rock = InclusionRock(mineralset = mineralset,
                     fluidset = fluidset,
                     total_porosity = phit,
                     rock_frame_method = 'dem')

# Define and add inclusions to the rock
rock.add_inclusion(Inclusion('SPHEROID', aspect_ratio = 0.12, content = brine), host = clay)
rock.add_inclusion(Inclusion('SPHEROID', aspect_ratio = 0.09),           host = calcite)
rock.add_inclusion(Inclusion('SPHEROID', aspect_ratio = 0.2),            host = quartz)
```

# Contribution

Contributions are highly welcome. In case you find any bug or issue in the codes, please share it with me to correct that. Let's improve it together.
