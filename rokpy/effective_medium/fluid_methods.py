"""
Fluid effect relations of effective medium theory

This module provides a comprehensive set of rock physics models for estimating how pore fluids affect
the elastic properties of porous rocks. It implements various theoretical approaches for fluid
substitution and frequency-dependent effects.
"""
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy.special import iv
from rokpy.effective_medium import BoundMethods
from rokpy.utilities import MultiEnum, biot_characteristic_frequency, reuss_intercept_porosity
from rokpy.conversions import velocity_to_modulus
if TYPE_CHECKING:
    from rokpy.materials import FluidSet


class FluidEffectMethods():
    class MethodName(MultiEnum):
        LowFrequency = 'gassmann', 'lowfreq'
        FrequencyDependant = 'dvorkin', 'full'

#Frequency dependant
    @staticmethod
    def biot (dry_bulk : np.ndarray, 
              dry_shear : np.ndarray, 
              mineralset_bulk : np.ndarray, 
              fluidset_bulk : np.ndarray, 
              porosity : np.ndarray, 
              frequency : np.ndarray, 
              mineralset_density : np.ndarray, 
              fluidset_density : np.ndarray, 
              viscosity_to_permeability, 
              tortuosity : np.ndarray):
        """
        Geertsma and Smit's (1961) approximation to frequency-dependant elastic moduli of Biot's (1956) model.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        fraquency: np.ndarray
            Frequency of measurement (Hz)
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        viscosity_to_permeability: np.ndarray
            Ratio of fluid viscosity (cp) to rock permeability (md)
        tortuosity : np.ndarray, optional
            Tortuosity parameter (alpha >= 1)
            Default is calculated from tortuosity(phi)
            
        Returns
        -------
        bulk : np.ndarray
            Frequency-dependant bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Frequency-dependant shear modulus of saturated rock (GPa)
            
            
        Notes
        -----
        - This model is the Geertsma-Smit (1961) approximation to Biot's theoretical formulas.
        - Low-frequency limiting properties by this model are same as Gassmann's relation
        - For most crustal rocks, squirt dispersion is comparable to or greater than Biot's dispersion,therefore, 
          it's recommended to use Mavko-Jizba squirt theory first to estimate high-frequency wet-frame moduli, then 
          substitute the results into Biot's equations.
        - Tortuosity parameter alpha >= 1, with typical values around 2-3 for sandstones
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Biot, M.A., 1956. Theory of propagation of elastic waves in a fluid-saturated porous solid. I. Low-frequency range. The Journal of the Acoustical Society of America, 28(2), pp.168-178.
        .. [3] Saxena, V., Krief, M. and Adam, L., 2018. Handbook of borehole acoustics and rock physics for reservoir characterization. Elsevier.
        .. [4] Geertsma, J. and Smit, D.C., 1961. Some aspects of elastic wave propagation in fluid-saturated porous solids. Geophysics, 26(2), pp.169-181.

        """
        dry_bulk = dry_bulk*1e9
        dry_shear = dry_shear*1e9
        mineralset_bulk = mineralset_bulk*1e9
        fluidset_bulk = fluidset_bulk*1e9
        mineralset_density = mineralset_density*1e3
        fluidset_density = fluidset_density*1e3
        fc = biot_characteristic_frequency(porosity, fluidset_density, viscosity_to_permeability)
        rho = mineralset_density*(1-porosity) + fluidset_density*porosity
        rho12 = (1-tortuosity)*porosity*fluidset_density
        rho11 = (1-porosity)*mineralset_density - (1-tortuosity)*porosity*fluidset_density
        rho22 = tortuosity*porosity*fluidset_density
        rho_l = (rho12+rho22)/porosity
        rho_c = rho22/(porosity**2)

        cb = 1/dry_bulk
        cr = 1/mineralset_bulk
        cl = 1/fluidset_bulk
        beta = cr/cb
        H = (1-beta)**2 / ((1-porosity-beta)*cr + porosity*cl) + \
            (beta/cr + 4/3*dry_shear)
        K = (1-beta) / ((1-porosity-beta)*cr + porosity*cl)
        L = 1 / ((1-porosity-beta)*cr + porosity*cl) 

        gamma_l = rho_l/rho
        gamma_c = rho_c/rho
        sigma_K = K/H
        sigma_L = L/H

        p_velocity = np.sqrt(H/rho * \
                             ((gamma_c + sigma_L - 2*gamma_l*sigma_K)**2 + (fc/frequency)**2) / \
                             ((gamma_c + sigma_L - 2*gamma_l*sigma_K)*(gamma_c - gamma_l**2) + (fc/frequency)**2))
                             #Geertsma and Smit, 1961 (Eq. 25)
        q = tortuosity*fluidset_density/porosity - 1j*(1e-3/1e-15)*viscosity_to_permeability/(2*np.pi*frequency)
        M = np.sqrt((rho*q-fluidset_density**2)/(dry_shear*q))
        s_velocity = 1/np.real(M) #Saxena et al, 2018 (Section 2.2.2.2)

        bulk, shear = velocity_to_modulus(p_velocity, s_velocity, rho/1000)
        return bulk, shear

    @staticmethod 
    def biot_squirt (dry_bulk : np.ndarray, 
                     dry_shear : np.ndarray, 
                     mineralset_bulk : np.ndarray, 
                     fluidset_bulk : np.ndarray, 
                     porosity : np.ndarray, 
                     frequency : np.ndarray, 
                     mineralset_density : np.ndarray, 
                     fluidset_density : np.ndarray, 
                     viscosity_to_permeability, 
                     squirt_flow_length : np.ndarray, 
                     tortuosity : np.ndarray, 
                     saturation=1.):
        """
        Biot-Squirt (BISQ) model for squirt-flow solid/fluid interactions. 
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        fraquency: np.ndarray
            Frequency of measurement (Hz)
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        viscosity_to_permeability: np.ndarray
            Ratio of fluid viscosity (cp) to rock permeability (md)
        squirt_flow_length: np.ndarray
            Characteristic squirt-flow length (in m)
            - It has to be either guessed (should have the same order of magnitude as the average grain size or 
            the average crack length) or adjusted by using an experimental measurement of velocity versus 
            frequency.
            Dvorkin et al (1994) proposed to estimate this quantity by matching the computed and measured 
            attenuation coefficients.
        tortuosity : np.ndarray, optional
            Tortuosity parameter (alpha >= 1)
            Default is calculated from tortuosity(phi)
        satiration: np.ndarray
            Total fluid saturation
            
        Returns
        -------
        p_velocity : np.ndarray
            Frequency-dependant p_velocity of saturated rock (m/s)
            
        Notes
        -----
        - This model considers both Biot dispersion and squirt dispersion mechanisms simultaneously at 
          apparently full saturation. The model is applicable to rocks at high pressure with compliant cracks 
          closed, i.e. where phisoft=0 and hence KdryhiP = Kdry.
        - This model returns p-velocity only.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        dry_bulk = dry_bulk*1e9
        dry_shear = dry_shear*1e9
        mineralset_bulk = mineralset_bulk*1e9
        fluidset_bulk = fluidset_bulk*1e9
        mineralset_density = mineralset_density*1e3
        fluidset_density = fluidset_density*1e3

        Rs = squirt_flow_length*np.sqrt(saturation)
        w = 2*np.pi*frequency
        Mdry = dry_bulk + 4/3*dry_shear
        F = 1/(1/fluidset_bulk + 1/(porosity*mineralset_bulk)*(1-porosity-dry_bulk/mineralset_bulk)) 
        gamma = 1-dry_bulk/mineralset_bulk 
        wc = 2*np.pi * biot_characteristic_frequency(porosity, fluidset_density, viscosity_to_permeability) 
        rho1 = (1-porosity)*mineralset_density 
        rho2 = porosity*fluidset_density 
        rhoa = -(1-tortuosity)*porosity*fluidset_density
        lamda = np.sqrt(fluidset_density*w*wc/F*1j)
        # lamda = np.sqrt(fluidset_density*w**2/F*((porosity+rhoa/fluidset_density)/porosity + 1j*wc/w))

        J0 = iv(0,lamda*Rs)
        J1 = iv(1,lamda*Rs)
        Fsq = F*(1 - 2*J1/(lamda*Rs*J0))

        A = porosity*Fsq*Mdry / rho2**2
        B = (Fsq*(2*gamma -porosity - porosity*rho1/rho2) - (Mdry + Fsq*(gamma**2)/porosity) * (1+rhoa/rho2 + 1j*wc/w))/rho2
        C = rho1/rho2 + (1 + rho1/rho2)*(rhoa/rho2 + 1j*wc/w)

        Y = -B/2/A - np.sqrt((B/2/A)**2 - C/A)

        p_velocity = 1/np.real(np.sqrt(Y))
        return p_velocity

    @staticmethod 
    def dvorkin (dry_bulk : np.ndarray, 
                 dry_shear : np.ndarray, 
                 mineralset_bulk : np.ndarray, 
                 fluidset_bulk : np.ndarray, 
                 porosity : np.ndarray, 
                 frequency : np.ndarray, 
                 mineralset_density : np.ndarray, 
                 fluidset_density : np.ndarray, 
                 high_pressure_dry_bulk : np.ndarray, 
                 Z = 0.001):
        """
        Dvorkin et al (1995) model for Effective fluid saturated rock at any frequencies incorporating the the effect of squirt dispersion.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        fraquency: np.ndarray
            Frequency of measurement (Hz)
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        high_pressure_dry_bulk: np.ndarray
            Effective bulk modulus of the dry rock at very high effective pressure when cracks are closed (GPa)
        Z : np.ndarray, optional
            Controling parametr to match the model with real data
            Start matching with Z = 0.001 and optimize Z until the resulting Ksat, Gsat, Qp and Qs match 
            the measured data
            
        Returns
        -------
        bulk : np.ndarray
            Frequency-dependant bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Frequency-dependant shear modulus of saturated rock (GPa)
            
        Notes
        -----
        - This is a extension of Mavko-Jizba model (`mavko_jizba`) for any given frequency.
        - This model in very low frequencies (freq = 0) reduces to Gassmann formulation
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Dvorkin, J., Mavko, G., and Nur, N., 1995, Squirt flow in fully saturated rocks. GEOPHYSICS 60: 97-107

        """
        dry_bulk *= 1e9
        dry_shear *= 1e9
        mineralset_bulk *= 1e9
        fluidset_bulk *= 1e9
        mineralset_density *= 1e3
        fluidset_density *= 1e3
        high_pressure_dry_bulk *= 1e9

        #Section 6.12
        #Step 1.
        Kmsd = 1/(1/mineralset_bulk - 1/high_pressure_dry_bulk + 1/dry_bulk)

        #Step 2.
        a0 = 1 - dry_bulk/mineralset_bulk
        Q0 = mineralset_bulk/(a0-porosity)
        F0 = 1/(1/fluidset_bulk + 1/porosity/Q0)        
        dP2ds = -1/( a0*(1 + porosity*dry_bulk/(a0**2*F0) ))

        #Step 3.
        a = 1 - Kmsd/mineralset_bulk
        xi = Z*np.sqrt(2*np.pi*frequency*1j)
        J0 = iv(0,xi)
        J1 = iv(1,xi)
        fxi = 2*J1/(xi*J0)
        # fxi = 2/J0 * jvp(1, xi)

        # IF THIS PART IS UNCOMMENTED, REMEMER TO INCLUDE THE CORRESPONDING ARGUMENTS IN FUNCTION DEFINITION
        # if not soft_porosity: #i.e. soft_porosity<<1 
        #     Z = squirt_flow_length*np.sqrt(viscosity_to_permeability*a/mineralset_bulk*(1e-3 / 9.869233e-16)) 
        #                             #Dvorkin et al, 1995, Pg. 100
        #                             #We change the Z to this formulation to make
        #                             #this function more comparable to 'biotsquirt'.
        #     Kms = (Kmsd + a*mineralset_bulk*(1-fxi))/(1 + a*fxi*dP2ds)
        # else: #(Devorkin et al, 1995)
        #     Q = mineralset_bulk/(a - soft_porosity)
        #     F = 1/(1/fluidset_bulk + 1/soft_porosity/Q)
        #     Z = squirt_flow_length*np.sqrt(viscosity_to_permeability*soft_porosity/F*(1e-3 / 9.869233e-16)) 
        #     Kms = (Kmsd + a**2*F/soft_porosity*(1-fxi))/(1 + a*fxi*dP2ds)

        Kms = (Kmsd + a*mineralset_bulk*(1-fxi))/(1 + a*fxi*dP2ds)

        #Step 4.
        Kmm = 1/(1/Kms + 1/high_pressure_dry_bulk - 1/mineralset_bulk)

        #Step 5.
        am = 1 - Kmm/Kms
        Kr = Kmm/(1+am*dP2ds)

        #Step 6.
        Ktms = Kmsd + a*mineralset_bulk*(1-fxi)
        Kmd = 1/(1/Ktms + 1/high_pressure_dry_bulk - 1/mineralset_bulk)
        Gmm = 1/(1/dry_shear - 4/15*(1/dry_bulk - 1/Kmd))

        #Step 7.
        rhob = porosity*fluidset_density + (1-porosity)*mineralset_density
        Vp = np.sqrt(np.real(Kr+4/3*Gmm)/rhob)
        Vs = np.sqrt(np.real(Gmm)/rhob)
        bulk, shear = velocity_to_modulus(Vp, Vs, rhob/1000)
        return bulk, shear

    @staticmethod 
    def mavko_jizba_biot(dry_bulk : np.ndarray, 
                         dry_shear : np.ndarray, 
                         mineralset_bulk : np.ndarray, 
                         bound_fluid_bulk, 
                         fluidset_bulk : np.ndarray, 
                         porosity : np.ndarray, 
                         soft_porosity, 
                         frequency : np.ndarray, 
                         mineralset_density : np.ndarray, 
                         fluidset_density : np.ndarray, 
                         high_pressure_dry_bulk : np.ndarray, 
                         viscosity_to_permeability, 
                         tortuosity : np.ndarray):
        """
        Frequency-dependant elastic moduli incorporating both Squirt and Biot flow (by combination of Mavko-Jizba and Biot model).
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        bound_fluid_bulk : np.ndarray
            Bulk modulus of the bound fluid (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Total porosity of rock
        soft_porosity : np.ndarray
            Porosity that is closed at very high effective pressure
        fraquency: np.ndarray
            Frequency of measurement (Hz)
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        high_pressure_dry_bulk: np.ndarray
            Effective bulk modulus of the dry rock at very high effective pressure when cracks are closed (GPa)
        viscosity_to_permeability: np.ndarray
            Ratio of fluid viscosity (cp) to rock permeability (md)
        tortuosity : np.ndarray, optional
            Tortuosity parameter (alpha >= 1)
            Default is calculated from tortuosity(phi)
            
        Returns
        -------
        bulk : np.ndarray
            Frequency-dependant bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Frequency-dependant shear modulus of saturated rock (GPa)
            
            
        Notes
        -----
        - For most crustal rocks the amount of squirt dispersion is comparable to or greater than Biot's
          dispersion and thus using Biot's theory alone  will lead to poor predictions of high-frequency 
          saturated velocities. Thus in order incorporate the effect of squirt dispersion, this code estimates
          the frame moduli by `mavko_jizba_squirt` method and then substitute the results (as dry frame moduli)
          into the `biot` relations. 
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Geertsma, J. and Smit, D.C., 1961. Some aspects of elastic wave propagation in fluid-saturated porous solids. Geophysics, 26(2), pp.169-181.

        """

        wetframe_bulk, wetframe_shear = FluidEffectMethods.mavko_jizba_wetframe(dry_bulk, dry_shear, mineralset_bulk, bound_fluid_bulk, high_pressure_dry_bulk, soft_porosity )
        bulk, shear = FluidEffectMethods.biot(wetframe_bulk, wetframe_shear, mineralset_bulk, fluidset_bulk, porosity-soft_porosity, frequency, mineralset_density, fluidset_density, viscosity_to_permeability, tortuosity)
        return bulk, shear
    
#High frequency
    @staticmethod 
    def biot_high (dry_bulk : np.ndarray, 
                   dry_shear : np.ndarray, 
                   mineralset_bulk : np.ndarray, 
                   fluidset_bulk : np.ndarray, 
                   porosity : np.ndarray, 
                   mineralset_density : np.ndarray, 
                   fluidset_density : np.ndarray, 
                   tortuosity : np.ndarray):
        """
        Biot's (1956) model of high-frequency limiting velocities and corresponding moduli
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            NOTE: Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        tortuosity : np.ndarray, optional
            Tortuosity parameter (alpha >= 1)
            Default is calculated from tortuosity(phi)

        Returns
        -------
        fast_high_pvelocity : np.ndarray
            Fast high-frequency limiting P-wave velocity (m/s)
            NOTE: This velocity corresponds to overall fluid and solid motions that are in-phase and is most 
            easily observed in the laboratory and the field.
        slow_high_pvelocity : np.ndarray
            Slow high-frequency limiting P-wave velocity (m/s)
            NOTE: Correspond to highly dissipative wave in which the overall solid and fluid motions are out of phase
        high_svelocity : np.ndarray
            Fast high-frequency limiting S-wave velocity (m/s)
            
        Notes
        -----
        - This model is based on Biot's (1956) theoretical formulas for the frequency-dependent seismic 
          velocities of saturated rocks in terms of the dry-rock properties. 
        - The low-frequency limiting approximation to Biot's (1956) frequency dependant velocities (Vp0,Vs0) 
          are same as those predicted by Gassmann's relation.
        - For most crustal rocks the amount of squirt dispersion (which is not  included in Biot's formulation)
          is comparable to or greater than Biot's dispersion, and thus using Biot's theory alone will lead to 
          poor predictions (overestimated velocities) of high-frequency saturated velocities. Therefore, it's  
          recommended to use the Mavko-Jizba squirt theory first to estimate the high-frequency wet-frame moduli 
          and then substitute them into Biot's equations.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        #Page 367
        rho12 = (1-tortuosity)*porosity*fluidset_density
        rho11 = (1-porosity)*mineralset_density - (1-tortuosity)*porosity*fluidset_density
        rho22 = tortuosity*porosity*fluidset_density
        rho = mineralset_density*(1-porosity) + fluidset_density*porosity

        P = ((1-porosity)*(1-porosity-dry_bulk/mineralset_bulk)*mineralset_bulk + porosity*mineralset_bulk*dry_bulk/fluidset_bulk) /            \
            (1-porosity-dry_bulk/mineralset_bulk + porosity*mineralset_bulk/fluidset_bulk) + 4/3*dry_shear 
        Q = (1-porosity-dry_bulk/mineralset_bulk)*porosity*mineralset_bulk /                                                                    \
            (1-porosity-dry_bulk/mineralset_bulk + porosity*mineralset_bulk/fluidset_bulk)
        R = (porosity**2)*mineralset_bulk / (1-porosity-dry_bulk/mineralset_bulk + porosity*mineralset_bulk/fluidset_bulk)

        Delta = P*rho22 + R*rho11 - 2*Q*rho12

        fast_high_pvelocity = 1000*np.sqrt((Delta + np.sqrt(Delta**2 - 4*(rho11*rho22-rho12**2)*(P*R-Q**2)) / (2*(rho11*rho22-rho12**2))))
        slow_high_pvelocity = 1000*np.sqrt((Delta - np.sqrt(Delta**2 - 4*(rho11*rho22-rho12**2)*(P*R-Q**2)) / (2*(rho11*rho22-rho12**2))))
        high_svelocity = 1000*np.sqrt(dry_shear/(rho-porosity*fluidset_density/tortuosity))
        return fast_high_pvelocity, slow_high_pvelocity, high_svelocity

    @staticmethod 
    def geertsma_smit_high  (dry_bulk : np.ndarray, 
                             dry_shear : np.ndarray, 
                             mineralset_bulk : np.ndarray, 
                             fluidset_bulk : np.ndarray, 
                             porosity : np.ndarray, 
                             mineralset_density: np.ndarray, 
                             fluidset_density : np.ndarray, 
                             tortuosity : np.ndarray):
        """
        Geertsma and Smit's (1961) approximation to Biot's high-frequency limiting P-wave velocity.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            NOTE: Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        mineralset_density : np.ndarray
            Mineral density (g/cc)
        fluidset_density : np.ndarray
            Fluid density (g/cc)
        tortuosity : np.ndarray, optional
            Tortuosity parameter (alpha >= 1)
            Default is calculated from tortuosity(phi)
            
        Returns
        -------
        fast_high_pvelocity : np.ndarray
            Fast high-frequency limiting P-wave velocity (m/s)
            NOTE: This velocity corresponds to overall fluid and solid motions that are in-phase and is most 
            easily observed in the laboratory and the field.
            
            
        Notes
        -----
        - This model is the Geertsma-Smit (1961) approximation to Biot's theoretical formulas.
        - This form predicts velocities that are too high (by about 3%–6%) compared with the actual high-
          frequency limit of Biot.
        - For most crustal rocks the amount of squirt dispersion (which is not  included in Biot's formulation)
          is comparable to or greater than Biot's dispersion, and thus using Biot's theory alone will lead to 
          poor predictions (overestimated velocities) of high-frequency saturated velocities. Therefore, it's  
          recommended to use the Mavko-Jizba squirt theory first to estimate the high-frequency wet-frame moduli 
          and then substitute them into Biot's equations.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Geertsma, J. and Smit, D.C., 1961. Some aspects of elastic wave propagation in fluid-saturated porous solids. Geophysics, 26(2), pp.169-181.

        """

        #Eq 6.1.12
        rho = mineralset_density*(1-porosity) + fluidset_density*porosity
        fast_high_pvelocity = 1000*np.sqrt(1/(mineralset_density*(1-porosity)+porosity*fluidset_density*(1-1/tortuosity)) * \
                                           ((dry_bulk+4/3*dry_shear) + (porosity*rho/fluidset_density/tortuosity + (1-dry_bulk/mineralset_bulk)*(1-dry_bulk/mineralset_bulk-2*porosity/tortuosity))/    \
                                            ((1-dry_bulk/mineralset_bulk-porosity)/mineralset_bulk + porosity/fluidset_bulk)))
        return fast_high_pvelocity

    @staticmethod 
    def mavko_jizba_wetframe (dry_bulk : np.ndarray,
                             dry_shear : np.ndarray,
                             mineralset_bulk : np.ndarray, 
                             fluidset_bulk : np.ndarray, 
                             high_pressure_dry_bulk : np.ndarray, 
                             soft_porosity : np.ndarray):
        """
        Mavko and Jizba (1991) model for Wet-frame moduli of rock, incorporating the squirt effect, at very high frequencies where the thinnest cracks are filled with fluid. 
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        high_pressure_dry_bulk: np.ndarray
            Effective bulk modulus of the dry rock at very high effective pressure when cracks are closed (GPa)
        soft_porosity : np.ndarray
            Porosity of the voids that are closed at very high effective pressure.
            NOTE: The volume of the soft porosity at any pressure is estimated as the difference between the 
            total porosity and the extrapolation of the high-pressure trend. (see Mavko and Jizba, 1991, Fig.4)
            
        Returns
        -------
        wetframe_bulk : np.ndarray
            Unrelaxed Bulk modulus of wet-frame of the rock (GPa) at very high frequencies
        wetframe_shear : np.ndarray
            Unrelaxed Shear modulus of wet-frame of the rock (GPa) at very high frequencies
            
        Notes
        -----
        - The results of this model are the properties of wet-frame of the rock (not the saturated rock) at very
          high frequencies. Therefore, one more step is needed to calculate the saturated rock properties.
        - This model accounts for the effect of pore heterogenity on the effective properties of rock, as a very
          high-frequency wave passes through the saturated rocks.  
        - For most crustal rocks the amount of squirt dispersion is comparable to or greater than Biot's
          dispersion and thus using Biot's theory alone  will lead to poor predictions of high-frequency 
          saturated velocities. Thus in order incorporate the effect of squirt dispersion, the frame moduli 
          estimated by this model are substituted (as dry frame moduli) into the `biot` relations. 

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Mavko, G., & Jizba, D., 1991. Estimating grain-scale fluid effects on velocity dispersion in rocks. GEOPHYSICS, 56(12), 1940-1949.

        """

        wetframe_bulk = 1/(1/high_pressure_dry_bulk + (1/fluidset_bulk - 1/mineralset_bulk)*soft_porosity)
        if np.any(wetframe_bulk < dry_bulk):
            warnings.warn("Calculated wet-frame bulk is less than dry frame bulk, which is unreasonable. Check the inputs (especially soft porosity or high-pressure bulk modulus).")
        wetframe_shear = 1/(1/dry_shear + 4/15*(1/wetframe_bulk - 1/dry_bulk))
        return wetframe_bulk, wetframe_shear
    
    @staticmethod 
    def mavko_jizba_gurevich(dry_bulk : np.ndarray, 
                             dry_shear : np.ndarray, 
                             mineralset_bulk : np.ndarray, 
                             fluidset_bulk : np.ndarray, 
                             high_pressure_dry_bulk : np.ndarray, 
                             soft_porosity ):
        """
        Gurevich's (2005) Extension of Mavko and Jizba model of Wet-frame moduli to highly compressible pore fluids
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        high_pressure_dry_bulk: np.ndarray
            Effective bulk modulus of the dry rock at very high effective pressure when cracks are closed (GPa)
        soft_porosity : np.ndarray
            Porosity of the voids that are closed at very high effective pressure.
            NOTE: The volume of the soft porosity at any pressure is estimated as the difference between the 
            total porosity and the extrapolation of the high-pressure trend. (see Mavko and Jizba, 1991, Fig.4)
            
        Returns
        -------
        wetframe_bulk : np.ndarray
            Unrelaxed Bulk modulus of wet-frame of the rock (GPa) at very high frequencies
        wetframe_shear : np.ndarray
            Unrelaxed Shear modulus of wet-frame of the rock (GPa) at very high frequencies

        See Also
        --------
        FluidEffectMethod.mavko_jizba_wetframe : Mavko-Jizba model of wet-frame


        Notes
        -----
        - The results of this model are the properties of wet-frame of the rock (not the saturated rock) at very
          high frequencies. Therefore, one more step is needed to calculate the saturated rock properties.
        - This model accounts for the effect of pore heterogenity on the effective properties of rock, as a very
          high-frequency wave passes through the saturated rocks.  
        - For most crustal rocks the amount of squirt dispersion is comparable to or greater than Biot's
          dispersion and thus using Biot's theory alone  will lead to poor predictions of high-frequency 
          saturated velocities. Thus in order incorporate the effect of squirt dispersion, the frame moduli 
          estimated by this model are substituted (as dry frame moduli) into the `biot` relations. 

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        wetframe_bulk = 1/(  1/high_pressure_dry_bulk + 1/( 1/(1/dry_bulk - 1/high_pressure_dry_bulk) + 1/((1/fluidset_bulk - 1/mineralset_bulk)*soft_porosity) )  )
        if np.any(wetframe_bulk < dry_bulk):
            warnings.warn("Calculated wet-frame bulk is less than dry frame bulk, which is unreasonable. Check the inputs (especially soft porosity or high-pressure bulk modulus).")
        wetframe_shear = 1/(1/dry_shear + 4/15*(1/wetframe_bulk - 1/dry_bulk))
        return wetframe_bulk, wetframe_shear

#Low Frequency    
    @staticmethod 
    def gassmann (dry_bulk : np.ndarray, 
                  dry_shear : np.ndarray, 
                  mineralset_bulk : np.ndarray, 
                  fluidset_bulk : np.ndarray, 
                  porosity : np.ndarray):
        """
        Effective low-frequency bulk modulus of fluid saturated rock.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            NOTE: Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        bulk : np.ndarray
            Low-frequency bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Low-frequency shear modulus of saturated rock (GPa)
            
        Notes
        -----
        - Gassmann's equations assume a homogeneous mineral modulus and statistical isotropy of the pore space 
          but is free of assumptions about the pore geometry, other than the pore space is well-connected.
        - Gassmann relation is valid only at sufficiently low frequencies such that the induced pore pressures 
          are equilibrated throughout the pore space (i.e., there is sufficient time for the pore fluid to flow
          and eliminate waveinduced pore-pressure gradients). Therefore, it works best for very low-frequency 
          in-situ seismic data (<100Hz) and may perform less well as ultrasonic measurements (~106 Hz).
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        RHS = dry_bulk/(mineralset_bulk-dry_bulk) + fluidset_bulk/(porosity*(mineralset_bulk-fluidset_bulk))                     
        bulk = RHS*mineralset_bulk/(1+RHS)
        shear = dry_shear
        return bulk, shear
        
    @staticmethod 
    def gassmann_biot (dry_bulk : np.ndarray,
                       dry_shear : np.ndarray, 
                       mineralset_bulk : np.ndarray, 
                       fluidset_bulk : np.ndarray, 
                       porosity : np.ndarray, 
                       biot_coef = None):
        """
        Effective low-frequency bulk modulus of fluid saturated rock using the Biot coefficient
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            NOTE: Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        porosity : np.ndarray
            Porosity
        biot_coef : np.ndarray
            Biot coefficient
            - If not available, it will be internally calculated in the code.
            
        Returns
        -------
        bulk : np.ndarray
            Low-frequency bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Low-frequency shear modulus of saturated rock (GPa)

        See Also
        --------
        utilities.biot_coef : Biot coefficient relation   

        Notes
        -----
        - Gassmann's equations assume a homogeneous mineral modulus and statistical isotropy of the pore space 
          but is free of assumptions about the pore geometry, other than the pore space is well-connected.
        - Gassmann relation is valid only at sufficiently low frequencies such that the induced pore pressures 
          are equilibrated throughout the pore space (i.e., there is sufficient time for the pore fluid to flow
          and eliminate waveinduced pore-pressure gradients). Therefore, it works best for very low-frequency 
          in-situ seismic data (<100Hz) and may perform less well as ultrasonic measurements (~106 Hz).
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """
        if not biot_coef:
            biot_coef = biot_coef(dry_bulk, mineralset_bulk)
        M = 1/((biot_coef-porosity)/mineralset_bulk + porosity/fluidset_bulk)
        bulk = dry_bulk + (biot_coef**2)*M
        return bulk, dry_shear
    
    @staticmethod 
    def brown_korringa  (dry_bulk : np.ndarray, 
                         dry_shear : np.ndarray,
                         mineralset_bulk : np.ndarray, 
                         fluidset_bulk : np.ndarray, 
                         porosity : np.ndarray, 
                         pore_incompressibility : np.ndarray):
        """
        Brown and Korringa (1975) fluid substitution in rocks with heterogeneous mineralogy.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
            NOTE: Either dry-frame moduli or high-frequency unrelaxed wet-frame
            moduli predicted by Mavko-Jizba squirt theory
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
        pore_incompressibility: np.ndarray
            Effective Pore Incompressibility (GPa)
            
        Returns
        -------
        bulk : np.ndarray
            Low-frequency bulk modulus of saturated rock (GPa)
        shear : np.ndarray
            Low-frequency shear modulus of saturated rock (GPa)
            
        Notes
        -----
        - If mineral is homogenous, `pore_incompressibility = mineralset_bulk` and Brown-Corringa relation 
          reduces to Gassmann's equation.
        - This model, just like Gassmann relation, is used to estimate the change of low-frequency elastic 
          moduli of porous media caused by a change of pore fluids. But unlike Gassmann, rock solid may be 
          heterogeneous (mixed).
        - Depending on how load-bearing the softest mineral is, this function may lead to different results.
        - Use 'bkporestiffness' to calculate KphiS and Ks, prior to using this code.

        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        RHS = dry_bulk/(mineralset_bulk-dry_bulk) + (pore_incompressibility/mineralset_bulk)*fluidset_bulk/(porosity*(pore_incompressibility-fluidset_bulk))
        bulk = mineralset_bulk*RHS/(1+RHS)
        return bulk, dry_shear

    @staticmethod 
    def inverse_gassmann (saturated_bulk : np.ndarray,
                          saturated_shear : np.ndarray,
                          mineralset_bulk : np.ndarray, 
                          fluidset_bulk : np.ndarray, 
                          porosity : np.ndarray):
        """
        Effective low-frequency bulk modulus of fluid Dry rock from saturated rock
                
        Parameters
        ----------
        saturated_bulk : np.ndarray
            Effective bulk modulus of the saturated rock (GPa)
        saturated_shear : np.ndarray
            Effective shear modulus of the saturated rock (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        dry_bulk : np.ndarray
            Low-frequency bulk modulus of dry rock (GPa)
        dry_shear : np.ndarray
            Low-frequency shear modulus of dry rock (GPa)
            
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        RHS = saturated_bulk/(mineralset_bulk-saturated_bulk) - fluidset_bulk/(porosity*(mineralset_bulk-fluidset_bulk))                           
        dry_bulk = RHS*mineralset_bulk/(1+RHS)
        return dry_bulk, saturated_shear

    @staticmethod 
    def inverse_anisotropic_gassmann(Saturated_voigt_matrix : np.ndarray, 
                                     mineralset_bulk : np.ndarray, 
                                     fluidset_bulk : np.ndarray, 
                                     porosity : np.ndarray):
        """
        Anisotropic properties of dry rock from saturated rock using anisotropic form of Gassmann equation. 
                
        Parameters
        ----------
        Saturated_voigt_matrix : np.ndarray
            6-by-6 Voigt matrix of saturated rock (size=[6 6 nsamples])
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        fluidset_bulk : np.ndarray
            Effective bulk modulus of the pore fluid (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        dry_voigt_matrix : np.ndarray
            6-by-6 Voigt matrix of dry rock (size=[6 6 nsamples])
            
        Notes
        -----
        - This function returns the 6-by-6 Voigt matrix of Dry rock, given the 6-by-6 Voigt matrix of Saturated rock
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        delta = np.array([1,1,1,0,0,0]) # corresponding to: ij(kl)= {11,22,33,23,13,12} or p(q) = { 1, 2, 3, 4, 5, 6}
        dry_voigt_matrix = np.zeros_like(Saturated_voigt_matrix)
        nsamples = Saturated_voigt_matrix.shape[2]
        for p in range (6):
            for q in range (6):
                cdrypq=np.reshape(  Saturated_voigt_matrix[p,q,:]                    ,(nsamples,1))
                cpaa = np.reshape(  np.sum(Saturated_voigt_matrix[p,0:3,:], axis=1)  ,(nsamples,1))
                cbbq = np.reshape(  np.sum(Saturated_voigt_matrix[0:3,q,:], axis=0)  ,(nsamples,1))
                cbbaa= np.reshape(  np.sum(np.sum(Saturated_voigt_matrix[0:3,0:3,:], axis=1), axis=0)  ,(nsamples,1))
                dry_voigt_matrix[p,q,:] = cdrypq  - (mineralset_bulk*delta(p)-cpaa/3) * (mineralset_bulk*delta(q)-cbbq/3) / (porosity*mineralset_bulk*(mineralset_bulk-fluidset_bulk)/fluidset_bulk - (mineralset_bulk-cbbaa/9))
        return dry_voigt_matrix

    @staticmethod 
    def gassmann_linear_substitution (primary_saturated_bulk : np.ndarray, 
                                      primary_saturated_shear : np.ndarray,
                                      mineralset_bulk : np.ndarray, 
                                      primary_fluidset_bulk : np.ndarray, 
                                      final_fluidset_bulk : np.ndarray,
                                      porosity : np.ndarray):
        """
        linear form of Gassmann's substitution (Mavko and Mukergi, 1995)
                
        Parameters
        ----------
        primary_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock before fluid substitution (GPa)
        primary_saturated_shear : np.ndarray
            Effective shear modulus of the rock before fluid substitution (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        primary_fluidset_bulk : np.ndarray
            Effective bulk modulus of the fluid before substitution (GPa)
        final_fluidset_bulk : np.ndarray
            Effective bulk modulus of the fluid after substitution (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        final_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock after fluid substitution (GPa)
        final_saturated_shear : np.ndarray
            Effective shear modulus of the rock after fluid substitution (GPa)
            
        Notes
        -----
        - This model just like ordinary Gassmann relation, computes the effective low-frequency bulk modulus of the rock saturated by a new fluid.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        intercept_porosity, initial_intercept_bulk = reuss_intercept_porosity(primary_saturated_bulk, mineralset_bulk, primary_fluidset_bulk, porosity)
        final_intercept_bulk = BoundMethods.reuss([mineralset_bulk,final_fluidset_bulk] , [1-intercept_porosity,intercept_porosity])
        final_saturated_bulk = primary_saturated_bulk + porosity/intercept_porosity * (final_intercept_bulk-initial_intercept_bulk)
        return final_saturated_bulk, primary_saturated_shear
    
    @staticmethod 
    def gassmann_substitution (primary_saturated_bulk : np.ndarray,
                               primary_saturated_shear : np.ndarray, 
                               mineralset_bulk : np.ndarray, 
                               primary_fluidset_bulk : np.ndarray, 
                               final_fluidset_bulk : np.ndarray,
                               porosity : np.ndarray):
        """
        Gassmann rock moduli after fluid substitution 
                
        Parameters
        ----------
        primary_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock before fluid substitution (GPa)
        primary_saturated_shear : np.ndarray
            Effective shear modulus of the rock before fluid substitution (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        primary_fluidset_bulk : np.ndarray
            Effective bulk modulus of the fluid before substitution (GPa)
        final_fluidset_bulk : np.ndarray
            Effective bulk modulus of the fluid after substitution (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        final_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock after fluid substitution (GPa)
        final_saturated_shear : np.ndarray
            Effective shear modulus of the rock after fluid substitution (GPa)
            
        Notes
        -----
        - Gassmann's equations assume a homogeneous mineral modulus and statistical isotropy of the pore space 
          but is free of assumptions about the pore geometry, other than the pore space is well-connected. 
        - Gassmann relation is valid only at sufficiently low frequencies such% that the induced pore pressures
          are equilibrated throughout the pore space (i.e., there is sufficient time for the pore fluid to flow
          and eliminate waveinduced pore-pressure gradients). Therefore, it works best for very low-frequency 
          in-situ seismic data (<100Hz) and may perform less well as frequencies increase toward sonic logging
          (~10^4 Hz) and laboratory ultrasonic measurements (~10^6 Hz)..
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        RHS = primary_saturated_bulk/(mineralset_bulk-primary_saturated_bulk) - primary_fluidset_bulk/(porosity*(mineralset_bulk-primary_fluidset_bulk)) + final_fluidset_bulk/(porosity*(mineralset_bulk-final_fluidset_bulk))
        final_saturated_bulk= RHS*mineralset_bulk/(1+RHS)
        return final_saturated_bulk, primary_saturated_shear
    
    @staticmethod 
    def gassmann_fluidset_substitution (primary_saturated_bulk : np.ndarray, 
                                        primary_saturated_shear : np.ndarray,
                                        mineralset_bulk : np.ndarray, 
                                        primary_fluidset: "FluidSet", 
                                        final_fluidset: "FluidSet",
                                        porosity : np.ndarray):
        """
        Gassmann rock moduli after fluid substitution 
                
        Parameters
        ----------
        primary_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock before fluid substitution (GPa)
        primary_saturated_shear : np.ndarray
            Effective shear modulus of the rock before fluid substitution (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        primary_fluidset : np.ndarray
            Fluidset before substitution (GPa)
        final_fluidset : np.ndarray
            Fluidset after substitution (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        final_saturated_bulk : np.ndarray
            Effective bulk modulus of the rock after fluid substitution (GPa)
        final_saturated_shear : np.ndarray
            Effective shear modulus of the rock after fluid substitution (GPa)
            
        Notes
        -----
        - Gassmann's equations assume a homogeneous mineral modulus and statistical isotropy of the pore space 
          but is free of assumptions about the pore geometry, other than the pore space is well-connected. 
        - Gassmann relation is valid only at sufficiently low frequencies such% that the induced pore pressures
          are equilibrated throughout the pore space (i.e., there is sufficient time for the pore fluid to flow
          and eliminate waveinduced pore-pressure gradients). Therefore, it works best for very low-frequency 
          in-situ seismic data (<100Hz) and may perform less well as frequencies increase toward sonic logging
          (~10^4 Hz) and laboratory ultrasonic measurements (~10^6 Hz)..
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """

        RHS = primary_saturated_bulk/(mineralset_bulk-primary_saturated_bulk) - primary_fluidset.bulk/(porosity*(mineralset_bulk-primary_fluidset.bulk)) + final_fluidset.bulk/(porosity*(mineralset_bulk-final_fluidset.bulk))
        final_saturated_bulk= RHS*mineralset_bulk/(1+RHS)
        return final_saturated_bulk, primary_saturated_shear
        
    @staticmethod 
    def marion_bound_average_substitution   (Primary_bulk: np.ndarray, 
                                             primary_shear : np.ndarray, 
                                             mineralset_bulk : np.ndarray,
                                             mineralset_shear : np.ndarray, 
                                             primary_porematerial_bulk : np.ndarray, 
                                             primary_porematerial_shear : np.ndarray, 
                                             final_porematerial_bulk : np.ndarray, 
                                             final_porematerial_shear : np.ndarray,
                                             porosity : np.ndarray):
        """
        Marion's Bounding Average Method to substitute pore-filling material in a rock. 
                
        Parameters
        ----------
        primary_bulk : np.ndarray
            Effective bulk modulus of the rock before pore-material substitution (GPa)
        primary_shear : np.ndarray
            Effective shear modulus of the rock before pore-material substitution (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        mineralset_shear : np.ndarray
            Shear modulus of the mineral (GPa)
        primary_porematerial_bulk : np.ndarray
            Effective bulk modulus of the pore-material before substitution (GPa)
        primary_porematerial_shear : np.ndarray
            Effective bulk modulus of the pore-material before substitution (GPa)
        final_porematerial_bulk : np.ndarray
            Effective bulk modulus of the pore-material after substitution (GPa)
        final_porematerial_shear : np.ndarray
            Effective shear modulus of the pore-material after substitution (GPa)
        porosity : np.ndarray
            Porosity
            
        Returns
        -------
        final_bulk : np.ndarray
            Effective bulk modulus of the rock after pore-material substitution (GPa)
        final_shear : np.ndarray
            Effective shear modulus of the rock after pore-material substitution (GPa)
            
        Notes
        -----
        - This method primarily heuristic and therefore needs to be tested empirically. The idea behind this 
          method is reasonable but not proven.
        - Marion and others (Marion and Nur, 1991; Marion et al., 1992) showed that this method works quite well
          for several examples: predicting water-saturated rock velocities from dry-rock velocities and 
          predicting frozen-rock (ice-filled) velocities from water-saturated velocities.
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.

        """        
        Kl1, Gl1 = BoundMethods.hashin_shtrikman_lower( mineralset_bulk, mineralset_shear, primary_porematerial_bulk, primary_porematerial_shear, porosity )
        Ku1, Gu1 = BoundMethods.hashin_shtrikman_upper( mineralset_bulk, mineralset_shear, primary_porematerial_bulk, primary_porematerial_shear, porosity )
        wK = (Primary_bulk-Kl1)/(Ku1-Kl1)
        wG = (primary_shear-Gl1)/(Gu1-Gl1)

        Kl2, Gl2 = BoundMethods.hashin_shtrikman_lower( mineralset_bulk, mineralset_shear, final_porematerial_bulk, final_porematerial_shear, porosity )
        Ku2, Gu2 = BoundMethods.hashin_shtrikman_upper( mineralset_bulk, mineralset_shear, final_porematerial_bulk, final_porematerial_shear, porosity )
        final_bulk = Kl2 + wK*(Ku2-Kl2)
        final_shear = Gl2 + wG*(Gu2-Gl2)
        return final_bulk, final_shear

