.. image::
   _static/rokpy_logo.png

What is rokpy?
================
``rokpy`` is a comprehensive object-oriented Python package for building and tuning rock-physics models.

**by: Mostafa Abbasi**

Introduction
============

``rokpy`` is a comprehensive python package for Rock-physical modeling that includes a wide variety of models and relations. the package follows 
an object-oriented programming (OOP) scheme for modeling the rock and its components. According to this scheme, a rock is object with many 
different attributes such as porosity, minerals, fluids, inclusions, etc. and also many different properties such as velocities, bulk and shear 
moduli, density, etc. All these properties are already included into a rock (or any of its components), therefore, It doesn't require the user 
to write code for any of these features. 

Once a rock model is build and it's hyper parameters such as pore shapes or fluid and mineral properties are fine-tuned for a given formation in 
a filed, one may use the model for another field. Therefore, using the ``rokpy``, you may **"Model here, apply there"**.

In addition to pre-designed models of rock and its components, these users may also use an extensive library of rock-physical relations and methods
to build their own models from the scratch. These methods are categorized into different modules such as **bound methods**, **inclusion methods**, 
**contact methods** and **fluid effect methods**. 

The package methods are mainly written on the basis of the formulations and descriptions given in *The Rock-Physics handbook (Mavko et al., 2020)*,
however, the methods are not limited to this book and many other key literatures are also reviewed to build the package. It worth noting that I 
have been tried to verify the codes by regenerating the plots in the book and and corresponding papers. 

.. note::
   Despite the breadth of ``rokpy``, this is a one-man project developed merely by myself during my spare times in few last years. Therefore, it's not unlikely to see some bugs or mistakes in the codes. Therefore, I'll appreciate if you contact me if you find any issue in the codes or if you have any recommendations.
   
   **Mostafa Abbasi**
   
   abbasi.mstfa@gmail.com 

Key Features
============

- Object-oriented design for rock-physics modeling
- An extensive library of rock-physics models and relations
- Effective medium theory implementations
- Empirical relations (Vernik, Gardner, Castagna, Han, etc.)
- AVO methods
- Fluid and mineral property databases
- Well-log visualization utilities


Installation
============

To install the package, simply use the following pip command in your terminal:

.. code-block:: bash

   pip install rokpy


Examples
========

The package has an ``examples`` directory containing some tutorials notebooks. This tutorials give you a good primary outlook to implement 
different models and codes. These notebooks also show you different capabilities of the codes beyond just modeling.

Don't limit yourself to these notebooks and also dive into the documentations to explore different classes, attributes and functions.

Contributing
============

Contributions are highly welcome. In case you find any bug or issue in the codes, please share it with me to correct that. Let's improve it together.


Documentation
===================

Follow the below links to read ``rokpy`` documentation:

.. toctree::
   :maxdepth: 2

   modules/index
   tutorials/index
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
