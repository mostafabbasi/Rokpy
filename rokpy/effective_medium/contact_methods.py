"""
Contact (granular) medium relations for granular materials

This module provides rock physics models for calculating the elastic properties of dry
granular materials (sands, sphere packs) under various loading conditions and cementation
patterns. The models are based on microscopic contact mechanics and effective medium theory
for predicting frame moduli from grain properties, porosity, and contact parameters.
"""

from typing import Literal, Tuple
import numpy as np
from rokpy import conversions
from rokpy import utilities
from rokpy.effective_medium.bound_methods import BoundMethods
from rokpy.utilities import MultiEnum


class ContactMethods():
    class ContactMethodName(MultiEnum):
        HertzMindlin            = 'hertz_mindlin', 'hm'
        SoftSand                = 'soft_sand', 'soft'
        StiffSand               = 'stiff_sand', 'stiff'
        IntermediateSand        = 'intermediate_sand', 'mid'
        IntermediateStiffSand   = 'intermediate_stiff_sand', 'midstiff'
        IntermediateCementedSand= 'intermediate_cemented_sand', 'midcemented'
        ContactCementedSand     = 'contact_cemented_sand', 'ccement'
        SurfaceCementedSand     = 'surface_cemented_sand', 'scement'
        Digby                   = 'digby'
        Jenkins                 = 'jenkins'
        RoughHydrostaticWalton  = 'walton_hydrostatic_rough'
        SmoothHydrostaticWalton = 'walton_hydrostatic_smooth'
        Johnson                 = 'johnson'
        Brandt                  = 'brandt'
        UniaxialWalton          = 'walton_uniaxial'



    @staticmethod
    def hertz_mindlin (grain_bulk : np.ndarray, 
                       grain_shear : np.ndarray, 
                       porosity : np.ndarray, 
                       contact_no : np.ndarray, 
                       pressure : np.ndarray, 
                       adhesion_coef : np.ndarray = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hertz-Mindlin model of a dry random pack of spheres.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        adhesion_coef : np.ndarray, optional
            Contacts adhesion coefficient . 
            Default is 1. Range: 0 (Frictionless contacts) to 1 (Perfect adhesion)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
            
        Notes
        -----
        - Hertz–Mindlin model can be used to describe the properties of precompacted granular 
          rocks.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
  
        """
        a2R = utilities.contact_ratio (pressure, grain_bulk, grain_shear, porosity, contact_no)
        Sn2R, St2R = utilities.relative_stiffness(grain_bulk, grain_shear, a2R, adhesion_coef)

        dry_bulk = contact_no*(1-porosity)/(12*np.pi)*Sn2R
        dry_shear = contact_no*(1-porosity)/(20*np.pi)*(Sn2R + 3/2*St2R)
        return dry_bulk , dry_shear
    
    @staticmethod
    def soft_sand (grain_bulk : np.ndarray, 
                   grain_shear : np.ndarray, 
                   porosity : np.ndarray, 
                   contact_no : np.ndarray, 
                   pressure : np.ndarray, 
                   uncemented_porosity : np.ndarray, 
                   adhesion_coef : np.ndarray = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft-sand or uncemented-sand model for a dry random pack of spheres.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        adhesion_coef : np.ndarray, optional
            Contacts adhesion coefficient . 
            Default is 1. Range: 0 (Frictionless contacts) to 1 (Perfect adhesion)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - This model is a dry sand model in which cement is deposited away from grain contacts.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
       
        """

        hertzmindlin_bulk, hertzmindlin_shear  = ContactMethods.hertz_mindlin(grain_bulk, grain_shear, uncemented_porosity, contact_no, pressure , adhesion_coef)
        phip = porosity/uncemented_porosity
        dry_bulk, dry_shear = BoundMethods.hashin_shtrikman_lower( grain_bulk, grain_shear, hertzmindlin_bulk, hertzmindlin_shear, phip )
        return dry_bulk, dry_shear
    
    @staticmethod
    def stiff_sand (grain_bulk : np.ndarray, 
                    grain_shear : np.ndarray, 
                    porosity : np.ndarray, 
                    contact_no : np.ndarray, 
                    pressure : np.ndarray, 
                    uncemented_porosity : np.ndarray, 
                    adhesion_coef : np.ndarray = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stiff-sand or contact cemented-sand model for a dry random pack of spheres.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        adhesion_coef : np.ndarray, optional
            Contacts adhesion coefficient . 
            Default is 1. Range: 0 (Frictionless contacts) to 1 (Perfect adhesion)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - This model is a dry sand model in which cement is deposited at grain contacts.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
    
        """

        hertzmindlin_bulk, hertzmindlin_shear  = ContactMethods.hertz_mindlin(grain_bulk, grain_shear, uncemented_porosity, contact_no, pressure , adhesion_coef)
        phip = porosity/uncemented_porosity
        dry_bulk, dry_shear = BoundMethods.hashin_shtrikman_upper( grain_bulk, grain_shear, hertzmindlin_bulk, hertzmindlin_shear, phip )
        return dry_bulk, dry_shear

    @staticmethod
    def intermediate_sand (grain_bulk : np.ndarray, 
                            grain_shear : np.ndarray, 
                            porosity : np.ndarray, 
                            contact_no : np.ndarray, 
                            pressure : np.ndarray, 
                            uncemented_porosity : np.ndarray, 
                            adhesion_coef : np.ndarray = 1, 
                            contact_cement_saturation : np.ndarray = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intermediate sand model between Soft- and Stiff-sand models for a dry random pack of spheres.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        adhesion_coef : np.ndarray, optional
            Contacts adhesion coefficient . 
            Default is 1. Range: 0 (Frictionless contacts) to 1 (Perfect adhesion)
        contact_cement_saturation : np.ndarray, optional
            Contact cement saturation as an averaging coefficient of soft bound . 
            Default is 0.5. Range: 0 (Soft-model) to 1 (Stiff-model)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - This is different from intermediate stiff-sand model ('intermediatestiffsand')
        - In the `intermediate_sand` model we use the Modified Hashin-Shtrikman's bounds with mineral 
          properties at phi=0 end and Hertz-Mindlin properties at the phi=phic. Note that this 
          is different from conventional use of modified Hashin-Shtrikman where the second 
          component is set at phi=1 and the value at the critical porosity is found along the 
          lower bound. This is why we didn't simply use `modified_hashin_shtrikman` in this code.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
   
        """
        
        bulk_soft, shear_soft = ContactMethods.soft_sand(grain_bulk, grain_shear, porosity, contact_no, pressure, uncemented_porosity, adhesion_coef)
        bulk_stiff, shear_stiff = ContactMethods.stiff_sand(grain_bulk, grain_shear, porosity, contact_no, pressure, uncemented_porosity, adhesion_coef)
        dry_bulk = BoundMethods.bounds_average(bulk_stiff, bulk_soft, contact_cement_saturation)
        dry_shear = BoundMethods.bounds_average(shear_stiff, shear_soft, contact_cement_saturation)
        return dry_bulk, dry_shear

    @staticmethod
    def intermediate_stiff_sand(grain_bulk : np.ndarray, 
                                grain_shear : np.ndarray, 
                                porosity : np.ndarray, 
                                contact_no : np.ndarray, 
                                pressure : np.ndarray, 
                                uncemented_porosity : np.ndarray, 
                                adhesion_coef : np.ndarray = 1., 
                                contact_cement_saturation : np.ndarray = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intermediate sand model between grain moduli and stiff-sand models for a dry random pack of spheres.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        adhesion_coef : np.ndarray, optional
            Contacts adhesion coefficient . 
            Default is 1. Range: 0 (Frictionless contacts) to 1 (Perfect adhesion)
        contact_cement_saturation : np.ndarray, optional
            Contact cement saturation (Vcc/phi0)
            Default is 0.5. Range: 0 (Soft-model) to 1 (Stiff-model)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - This model lies somewhere between the soft-sand (uncemented) and stiff-sand (contact 
          cemented) models depending on the cement saturation.
        - The easiest way to generate the intermediate stiff-sand model is by simply increasing 
          the coordination number in the soft-sand model, however in this code I use the idea 
          given by Hossain and McGregor (2014) where they assume that the lower end point of 
          modified Hashin-Shtrikman is migrated from the critical porosity to a lower porocity 
          as follows:

          .. math::
              {\\phi_0}_{new} = \\phi_0 * (1-Sc)
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Hossain, Z., & MacGregor, L., 2014. Advanced rock-physics diagnostic analysis: A new method for cement quantification. The Leading Edge, 33(3), 310–316.

        """
        cemented_porosity = contact_cement_saturation*uncemented_porosity
        stiff_bulk, stiff_shear = ContactMethods.stiff_sand(grain_bulk, grain_shear, uncemented_porosity-cemented_porosity, contact_no, pressure, uncemented_porosity, adhesion_coef)

        phip = porosity/(uncemented_porosity-cemented_porosity)
        dry_bulk, dry_shear = BoundMethods.hashin_shtrikman_upper( grain_bulk, grain_shear, stiff_bulk, stiff_shear, phip )

        return dry_bulk, dry_shear

    @staticmethod
    def digby ( grain_bulk : np.ndarray, 
                grain_shear : np.ndarray, 
                porosity : np.ndarray, 
                contact_no : np.ndarray, 
                pressure : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Digby's (1981) model for a dry pack of infinitely rough identical spheres under a uniaxial pressure.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
       
        """
        
        grain_poisson = conversions.modulus_to_poisson(grain_bulk, grain_shear)
        a2R = utilities.contact_ratio(pressure, grain_bulk, grain_shear, porosity, contact_no)
        term4 = -(3*np.pi/2) * (1-grain_poisson)/(contact_no*(1-porosity))*(pressure/grain_shear)

        d = np.atleast_1d(term4)
        p = np.atleast_2d(np.hstack([np.full_like(a2R, 1), np.full_like(a2R, 0), a2R, term4]))
        for k in range(np.size(term4)):
            allroots = np.roots(p[k,:])
            d[k] = np.real(allroots[2])             

        b2R = np.sqrt( d**2 + a2R**2)

        Sn2R = 4*grain_shear*b2R / (1-grain_poisson)
        St2R = 8*grain_shear*a2R / (2-grain_poisson)

        dry_bulk = 1/12/np.pi * contact_no*(1-porosity)*Sn2R
        dry_shear = 1/20/np.pi * contact_no*(1-porosity)*(Sn2R+3/2*St2R)

        return dry_bulk, dry_shear
    
    @staticmethod
    def jenkins (grain_bulk : np.ndarray, 
                 grain_shear : np.ndarray, 
                 porosity : np.ndarray, 
                 contact_no : np.ndarray, 
                 pressure : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jenkins et al (2005) model for a dry random packing of identical frictionless spheres under a uniaxial pressure. 
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - Their approach differs from that of Digby (1981), Walton (1987), and Hertz−Mindlin in 
          that under applied deviatoric strain, the particle motion of each sphere relative to 
          its neighbors is allowed to deviate from the mean homogeneous strain field of a 
          corresponding homogeneous effective medium. The additional degrees of freedom of 
          particle motion result in calculated shear moduli that are smaller than predicted by 
          the previously mentioned models.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
     
        """
        psi = contact_no/3
        w1 = (166-11*contact_no)/128
        wt1 = (38-11*contact_no)/128
        w2 = -(contact_no+14)/128
        a1 = (19*contact_no-22)/48
        a2 = (22-3*contact_no)/16
        at2 = (18-9*contact_no)/48

        k1 = -(a1*wt1 + at2*wt1 + 2*at2*w2) + wt1 - (0.52*(contact_no-2)*(contact_no-4) \
            + 0.10*contact_no*(contact_no-2) - 0.13*contact_no*(contact_no-4) - 0.01*contact_no**2)/(16*np.pi)
        k2 = -a1*w2 + w2 + (0.44*(contact_no-2)*(contact_no-4) - 0.24*contact_no*(contact_no-2) \
            - 0.11*contact_no*(contact_no-4) - 0.14*contact_no**2)/(16*np.pi)
        k3 = -(a1*w2 + at2*w2) + w2 - (0.44*(contact_no-2)*(contact_no-4) - 0.42*contact_no*(contact_no-2) \
            - 0.11*contact_no*(contact_no-4) + 0.04*contact_no**2)/(16*np.pi)

        eta1 = -a1**2 + a1 + (1.96*(contact_no-2)*(contact_no-4) + 3.3*contact_no*(contact_no-2) \
            + 0.49*contact_no*(contact_no-4) + 0.32*contact_no**2)/(16*np.pi)
        eta2 = -(2*a1*at2 + at2**2) + at2 - (2.16*(contact_no-2)*(contact_no-4) + 2.30*contact_no*(contact_no-2) \
            +0.54*contact_no*(contact_no-4) - 0.06*contact_no**2)/(16*np.pi)

        z1 = eta1*w1 + eta2*w1 + 2*eta2*w2
        z2 = eta1*w2
        z3 = eta1*w2 + eta2*w2

        a2R = utilities.contact_ratio(pressure, grain_bulk, grain_shear, porosity, contact_no)
        Sn2R, _ = utilities.relative_stiffness(grain_bulk, grain_shear, a2R, 1)
        # NOTE: Stn = Sn/2 where Sn is Hertz-Mindlin Stiffness

        dry_shear = contact_no*(1-porosity)/(5*np.pi) * (1/4*Sn2R)*(1 - 2*((w1+2*w2)/psi \
            - (k1+2*k2)/psi**2 + (z1+2*z2)/psi**3))
        dry_lame = contact_no*(1-porosity)/(5*np.pi) * (1/4*Sn2R)*(1 - 2*((w1+7*w2)/psi \
            - (k1+2*k2+5*k3)/psi**2 + (z1+2*z2+5*z3)/psi**3))
        dry_bulk = (dry_lame + 2/3*dry_shear)
        return dry_bulk, dry_shear
    
    @staticmethod
    def walton_hydrostatic (grain_bulk : np.ndarray, 
                            grain_shear : np.ndarray, 
                            porosity : np.ndarray, 
                            contact_no : np.ndarray, 
                            pressure : np.ndarray, 
                            friction: Literal['rough', 'smooth']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walton's (1987) model for a dry pack of identical spheres under a hydrostatic pressure. 
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
        friction : str
            Grain friction flag:
            - 'smooth': smooth frictionless grain contacts
            - 'rough': rough high friction grain contacts

            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - Under a hydrostatic pressure, such a medium is isotropic. 
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
     
        """
 
        Lm = conversions.modulus_to_Lame(grain_bulk, grain_shear)

        A = 1/4/np.pi * (1./grain_shear - 1./(grain_shear+Lm))
        B = 1/4/np.pi * (1./grain_shear + 1./(grain_shear+Lm))

        match friction:
            case 'rough':
                dry_bulk = 1/10 * ((3/np.pi**4) * ((1-porosity)*contact_no/B)**2 * pressure)**(1/3)
                dry_shear = 3/5 * dry_bulk * (5*B+A)/(2*B+A)

            case 'smooth':
                dry_shear = 1/10 * ((3/np.pi**4) * ((1-porosity)*contact_no/B)**2 * pressure)**(1/3)
                dry_bulk = 5/3 * dry_shear

        return dry_bulk, dry_shear
    
    @staticmethod
    def walton_hydrostatic_rough (grain_bulk : np.ndarray, 
                                  grain_shear : np.ndarray, 
                                  porosity : np.ndarray, 
                                  contact_no : np.ndarray, 
                                  pressure : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walton's model for a dry pack of identical rough spheres under a hydrostatic pressure. 
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - Under a hydrostatic pressure, such a medium is isotropic. 
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
     
        """
        dry_bulk, dry_shear = ContactMethods.walton_hydrostatic(grain_bulk, grain_shear, porosity, contact_no, pressure , 'rough')
        return dry_bulk, dry_shear

    @staticmethod
    def walton_hydrostatic_smooth (grain_bulk : np.ndarray, 
                                   grain_shear : np.ndarray, 
                                   porosity : np.ndarray, 
                                   contact_no : np.ndarray, 
                                   pressure : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walton's model for a dry pack of identical smooth spheres under a hydrostatic pressure. 
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Hydrostatic confining pressure (GPa) 
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus

        Notes
        -----
        - Under a hydrostatic pressure, such a medium is isotropic.            
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
 
        """
        dry_bulk, dry_shear = ContactMethods.walton_hydrostatic(grain_bulk, grain_shear, porosity, contact_no, pressure , 'smooth')
        return dry_bulk, dry_shear
    
    @staticmethod
    def walton_uniaxial (grain_bulk : np.ndarray, 
                         grain_shear : np.ndarray, 
                         porosity : np.ndarray, 
                         contact_no : np.ndarray, 
                         pressure : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Walton's model for a dry pack of infinitely rough identical spheres under a uniaxial pressure. 
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        pressure : np.ndarray
            Uniaxial pressure  (Same unit as moduli)
            
        Returns
        -------
        (c11, c33, c44, c66, c12, c13) : tuple
            Effective Voigt matrix elements of dry rock
            
        Notes
        -----
        - Under a uniaxial pressure, such a medium is transversely isotropic.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
         
        """
        
        grain_lame = conversions.modulus_to_Lame(grain_bulk,grain_shear)

        A = 1/4/np.pi * (1./grain_shear - 1./(grain_shear+grain_lame))
        B = 1/4/np.pi * (1./grain_shear + 1./(grain_shear+grain_lame))

        e = (24*np.pi**2 * B*(2*B+A)/((1-porosity)*A*contact_no) * pressure)**(1/3)
        a = (1/32/np.pi**2) * (1-porosity)*contact_no*e / B
        b = (1/32/np.pi**2) * (1-porosity)*contact_no*e / (2*B+A)

        c11 = 3*(a+2*b)
        c12 = a-2*b
        c13 = 2*c12
        c33 = 8*(a+b)
        c44 = a+7*b
        c66 = 1/2*(c11-c12)

        return c11, c33, c44, c66, c12, c13

    @staticmethod
    def cemented_sand (grain_bulk : np.ndarray, 
                       grain_shear : np.ndarray, 
                       porosity : np.ndarray, 
                       contact_no : np.ndarray, 
                       uncemented_porosity : np.ndarray, 
                       cement_bulk : np.ndarray, 
                       cement_shear : np.ndarray, 
                       cementat: Literal['contact', 'surface']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dry cemented-sand model where cement is deposited at or away from grain contacts.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        contact_no : np.ndarray
            Coordination number 
        cement_bulk : np.ndarray
            Bulk modulus of cement
        cement_shear : np.ndarray
            Shear modulus of cement
        cementat : str
            Location at which cement is deposited:
            'contact': If cements are deposited at grains contacts.
            'surface': If cements are deposited at grain surfaces away from contacts.            

        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - The cement is elastic and its properties may differ from those of the spheres (Dvorkin and Nur, 1996).
        - This model can provide a reliable tool to diagnose the cement type.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
       
        """

        match cementat.lower():
            case 'contact':
                a2R = 2*(1/3 * (uncemented_porosity-porosity)/(contact_no*(1-uncemented_porosity)))**0.25
            case 'surface':
                a2R = (2/3 * (uncemented_porosity-porosity)/(1-uncemented_porosity))**0.5

        Mc = cement_bulk + 4/3*cement_shear
        grain_poisson = conversions.modulus_to_poisson(grain_bulk, grain_shear)
        cement_poisson = conversions.modulus_to_poisson(cement_bulk, cement_shear)

        Lambda_t = 1/np.pi * cement_shear/grain_shear
        Lambda_n = 2/np.pi * cement_shear/grain_shear * (1-grain_poisson)*(1-cement_poisson)/(1-2*cement_poisson)

        An = -0.024153   *Lambda_n**(-1.36460)
        Bn =  0.20405    *Lambda_n**(-0.89008)
        Cn =  0.00024649*Lambda_n**(-1.98640)

        At = -0.01 *(2.26  *grain_poisson**2 + 2.07  *grain_poisson + 2.300) *Lambda_t**(0.079  *grain_poisson**2 + 0.1754*grain_poisson - 1.3420)
        Bt =        (0.0573*grain_poisson**2 + 0.0937*grain_poisson + 0.202) *Lambda_t**(0.027  *grain_poisson**2 + 0.0529*grain_poisson - 0.8765)
        Ct = 0.0001*(9.654 *grain_poisson**2 + 4.9450*grain_poisson + 3.100) *Lambda_t**(0.01867*grain_poisson**2 + 0.4011*grain_poisson - 1.8186)

        Sn = An*(a2R**2) + Bn*a2R + Cn
        St = At*(a2R**2) + Bt*a2R + Ct

        dry_bulk = 1/6 * contact_no*(1-uncemented_porosity)*Mc*Sn
        dry_shear = 3/5*dry_bulk + 3/20 *contact_no*(1-uncemented_porosity)*cement_shear*St

        return dry_bulk, dry_shear
    
    @staticmethod
    def contact_cemented_sand (grain_bulk : np.ndarray, 
                               grain_shear : np.ndarray, 
                               porosity : np.ndarray, 
                               contact_no : np.ndarray, 
                               uncemented_porosity : np.ndarray, 
                               cement_bulk : np.ndarray, 
                               cement_shear : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dry cemented-sand model where cement is deposited at grain contacts.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        contact_no : np.ndarray
            Coordination number 
        cement_bulk : np.ndarray
            Bulk modulus of cement
        cement_shear : np.ndarray
            Shear modulus of cement
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - The cement is elastic and its properties may differ from those of the spheres (Dvorkin and Nur, 1996).
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        
        """
        dry_bulk, dry_shear = ContactMethods.cemented_sand(grain_bulk, grain_shear, porosity, contact_no, uncemented_porosity, cement_bulk, cement_shear, 'contact')
        return dry_bulk, dry_shear

    @staticmethod
    def surface_cemented_sand (grain_bulk : np.ndarray, 
                               grain_shear : np.ndarray, 
                               porosity : np.ndarray, 
                               contact_no : np.ndarray, 
                               uncemented_porosity : np.ndarray, 
                               cement_bulk : np.ndarray, 
                               cement_shear : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dry cemented-sand model where cement is deposited on grain surface (away from contacts).
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains 
        grain_shear : np.ndarray
            Shear modulus of solid grains 
        porosity : np.ndarray
            Porosity 
        contact_no : np.ndarray
            Coordination number 
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity) 
        cement_bulk : np.ndarray
            Bulk modulus of cement
        cement_shear : np.ndarray
            Shear modulus of cement

        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - The cement is elastic and its properties may differ from those of the spheres (Dvorkin and Nur, 1996).
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
         
        """
        dry_bulk, dry_shear = ContactMethods.cemented_sand(grain_bulk, grain_shear, porosity, contact_no, uncemented_porosity, cement_bulk, cement_shear, 'surface')
        return dry_bulk, dry_shear
    
    @staticmethod
    def intermediate_cemented_sand (grain_bulk : np.ndarray, 
                                    grain_shear : np.ndarray, 
                                    porosity : np.ndarray, 
                                    contact_no : np.ndarray, 
                                    uncemented_porosity : np.ndarray, 
                                    cement_bulk : np.ndarray, 
                                    cement_shear : np.ndarray, 
                                    contact_cement_saturation : np.ndarray = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intermediate cemented-sand model between contact- and surface-cemented sand models.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains
        grain_shear : np.ndarray
            Shear modulus of solid grains
        porosity : np.ndarray
            Porosity
        contact_no : np.ndarray
            Coordination number
        uncemented_porosity : np.ndarray
            Uncemented porosity (critical porosity)
        cement_bulk : np.ndarray
            Bulk modulus of cement
        cement_shear : np.ndarray
            Shear modulus of cement
        contact_cement_saturation : np.ndarray, optional
            Contact cement saturation as an averaging coefficient of soft bound
            Default is 0.5. Range: 0 (surface-cemented) to 1 (contact-cemented)
            
        Returns
        -------
        dry_bulk  : np.ndarray
            Effective dry intermediate-sand bulk modulus
        dry_shear : np.ndarray
            Effective dry intermediate-sand shear modulus
            
        Notes
        -----
        - This is a conceptual Voigt-Reuss-Hill average model of contact- and 
          surface-cemented sand model.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
  
        """
        bulk_soft, shear_soft = ContactMethods.surface_cemented_sand(grain_bulk, grain_shear, porosity, contact_no, uncemented_porosity, cement_bulk, cement_shear)
        bulk_stiff, shear_stiff = ContactMethods.contact_cemented_sand(grain_bulk, grain_shear, porosity, contact_no, uncemented_porosity, cement_bulk, cement_shear)
        dry_bulk = BoundMethods.bounds_average(bulk_stiff, bulk_soft, contact_cement_saturation)
        dry_shear = BoundMethods.bounds_average(shear_stiff, shear_soft, contact_cement_saturation)
        return dry_bulk, dry_shear

    @staticmethod
    def johnson (grain_bulk : np.ndarray, 
                 grain_shear : np.ndarray, 
                 porosity : np.ndarray, 
                 contact_no : np.ndarray, 
                 hydrostatic_strain, 
                 uniaxial_strain) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Johnson et al (1998)'s model for the nonlinear elasticty of granular dry sphere packs.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains
        grain_shear : np.ndarray
            Shear modulus of solid grains
        porosity : np.ndarray
            Porosity
        contact_no : np.ndarray
            Coordination number
        hydrostatic_strain : np.ndarray
            Absolute value of Hydrostatic strain
        uniaxial_strain : np.ndarray
            Absolute value of Uniaxial strain

            
        Returns
        -------
        (c11, c33, c44, c66, c12, c13) : tuple
            Effective Voigt matrix elements of dry rock
            
        Notes
        -----
        - This model generalize the Walton's models (which is given just for two limiting cases 
          of hydrostatic and uniaxial pressure) based on no-slip Hertz-Mindlin theory of 
          grain-to-grain contacts.
        - When the strain is a combination of hydrostatic and uniaxial compression, the sphere 
          pack exhibits a transversely isotropic symmetry. This a stress-induced anisotropy.
        - When strain is a combination of hydrostatic and uniaxial compression (along 3rd axis), 
          total strain is as follows:

          .. math::
              \\epsilon_{ij} =   \\epsilon * \\delta_{ij}   +   \\epsilon_3 * \\delta_{i3}*\\delta_{j3}

        - Results are consistent with the Walton model in the limiting cases of pure hydrostatic 
          strain (ep3 --> 0) and pure uniaxial strain (ep --> 0)
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Johnson, D.L., Schwartz, L.M., Elata, D., Berryman, J.G., Hornby, B. and Norris, A.N., 1998. Linear and nonlinear elasticity of granular media: Stress-induced anisotropy of a random sphere pack.

        """

        num = conversions.modulus_to_poisson(grain_bulk, grain_shear)

        Ct = 8*grain_shear/(2-num)
        Cn = 4*grain_shear/(1-num)
        Cw = 4/np.pi * (1./Ct - 1./Cn)
        Bw = 2/np.pi * (1./Cn)

        a = np.sqrt(hydrostatic_strain/uniaxial_strain)
        gamma = 3/32*contact_no*Cn*Ct*(1-porosity)*np.sqrt(hydrostatic_strain)

        I0a = 1/2 * ( (1+a**2)**(1/2) +   (a**2)*np.log((1+np.sqrt(1+a**2))/a) )
        I2a = 1/4 * ( (1+a**2)**(3/2) -   (a**2)*I0a)
        I4a = 1/6 * ( (1+a**2)**(3/2) - 3*(a**2)*I2a)

        c11 = gamma/a *(  2*Bw*(I0a-I2a) + 3/4*Cw*(I0a-2*I2a + I4a))
        c13 = gamma/a *(                       Cw*(I2a-I4a)        )
        c33 = gamma/a *(  4*Bw*I2a       +   2*Cw*I4a              )
        c44 = gamma/a *(1/2*Bw*(I0a+I2a) +     Cw*(I2a-I4a)        )
        c66 = gamma/a *(    Bw*(I0a-I2a) + 1/4*Cw*(I0a-2*I2a+I4a)  )
        c12 = c11 - 2*c44
        return c11, c33, c44, c66, c12, c13
    
    @staticmethod
    def brandt (grain_bulk : np.ndarray, 
                grain_shear : np.ndarray, 
                porosity : np.ndarray, 
                pressure : np.ndarray, 
                fluidset_bulk : np.ndarray) -> np.ndarray:
        """
        Brandt's Model of dry random pack of spheres of Different sizes.
        
        Parameters
        ----------
        grain_bulk : np.ndarray
            Bulk modulus of solid grains
        grain_shear : np.ndarray
            Shear modulus of solid grains
        porosity : np.ndarray
            Porosity
        pressure : np.ndarray
            Hydrostatic confining pressure (same unit as moduli)
        fluidset_bulk : np.ndarray
            Bulk modulus of fluid
            
        Returns
        -------
        dry_bulk : np.ndarray
            Effective dry intermediate-sand bulk modulus 
            
        Notes
        -----
        Brandt's model consists of randomly packed spheres of identical mechanical properties 
        but different sizes subjected to external and internal hydrostatic pressures. The 
        effective pressure P is the difference between these two pressures.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
    
        """
        grain_young = conversions.modulus_to_young(grain_bulk, grain_shear)
        grain_poisson = conversions.modulus_to_poisson(grain_bulk, grain_shear)

        z = fluidset_bulk**(3/2)*(1-grain_poisson**2)/(grain_young*np.sqrt(pressure))
        Z = (1 + 30.75*z)**(5/3) / (1 + 46.13*z)
        dry_bulk = 2*pressure**(1/3) / (9*porosity)*(grain_young/(1.75*(1-grain_poisson**2)))**(2/3) * Z - 1.5*pressure*Z
        return dry_bulk
