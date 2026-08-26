"""
Fluid-effect Q-factor relations of fluid effect

This module implements theoretical models for estimating seismic quality factors (Q factors)
and attenuation coefficients in porous rocks. Quality factors quantify the dissipation of
seismic wave energy during propagation, providing insights into rock heterogeneity,
fluid mobility, and reservoir characterization.

"""

from . import utilities
import numpy as np
from scipy.special import jv

    
class QualityFactorMethods():
    @staticmethod 
    def biot_squirt     (dry_bulk : np.ndarray, 
                         dry_shear, 
                         mineralset_bulk, 
                         fluidset_bulk, 
                         porosity, 
                         frequency, 
                         mineralset_density, 
                         fluidset_density, 
                         viscosity_to_permeability, 
                         squirt_flow_length, 
                         tortuosity, 
                         saturation=1):
        """
        Biot-Squirt (BISQ) model P-wave quality factor. 
                
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
            It has to be either guessed (should have the same order of magnitude as the average grain size or 
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
        p_qfactor: np.ndarray
            Frequency-dependant P-wave quality factor of saturated rock
        p_attenuation: np.ndarray
            Frequency-Dependant p-wave attenuation factor
            
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
        wc = 2*np.pi * utilities.biot_characteristic_frequency(porosity, fluidset_density, viscosity_to_permeability) 
        rho1 = (1-porosity)*mineralset_density 
        rho2 = porosity*fluidset_density 
        rhoa = -(1-tortuosity)*porosity*fluidset_density
        lamda = np.sqrt(fluidset_density*w*wc/F*1j)

        J0 = jv(0,lamda*Rs)
        J1 = jv(1,lamda*Rs)
        Fsq = F*(1 - 2*J1/(lamda*Rs*J0))

        A = porosity*Fsq*Mdry / (rho2**2)
        B = (Fsq*(2*gamma - porosity - porosity*rho1/rho2) - (Mdry + Fsq*(gamma**2)/porosity)* (1+ rhoa/rho2 + 1j*wc/w))/rho2
        C = rho1/rho2 + (1 + rho1/rho2)*(rhoa/rho2 + 1j*wc/w)

        Y = -B/A/2 - np.sqrt((B/A/2)**2 - C/A)

        Vp = 1/np.real(np.sqrt(Y))
        p_attenuation = w*np.imag(np.sqrt(Y))
        p_qfactor = 1/abs((2*p_attenuation*Vp/w))
        return p_qfactor, p_attenuation

    @staticmethod 
    def dvorkin( dry_bulk, 
                dry_shear,  
                mineralset_bulk, 
                fluidset_bulk, 
                porosity, 
                frequency, 
                mineralset_density, 
                fluidset_density, 
                high_pressure_dry_bulk, 
                viscosity_to_permeability, 
                squirt_flow_length):
        """
        Dvorkin et al (1991) model of P- and S-quality factor for Effective fluid saturated rock at any frequencies incorporating the the effect of squirt dispersion.
                
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
        p_qfactor : np.ndarray
            Frequency-dependant P-wave Quality Factor of saturated rock (GPa)
        s_qfactor : np.ndarray
            Frequency-dependant S-wave Quality Factor modulus of saturated rock (GPa)
            
        Notes
        -----
        - This is a extension of Mavko-Jizba model (`mavko_jizba`) for any given frequency.
        - 
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Dvorkin, J., Mavko, G., and Nur, N., 1995, Squirt flow in fully saturated rocks. GEOPHYSICS 60: 97-107

        """

        dry_bulk = dry_bulk*1e9
        dry_shear = dry_shear*1e9
        mineralset_bulk = mineralset_bulk*1e9
        fluidset_bulk = fluidset_bulk*1e9
        mineralset_density = mineralset_density*1e3
        fluidset_density = fluidset_density*1e3
        high_pressure_dry_bulk = high_pressure_dry_bulk*1e9

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
        Z = squirt_flow_length*np.sqrt(viscosity_to_permeability*a/mineralset_bulk*(1e-3 / 9.869233e-16)) 
                                    #Dvorkin et al, 1995, Pg. 100
                                    #We change the Z to this formulation to make
                                    #this function more comparable to 'biotsquirt'.
        xi = Z*np.sqrt(2*np.pi*frequency*1j)
        J0 = jv(0,xi)
        J1 = jv(1,xi)
        fxi = 2*J1/(xi*J0)
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
        Qp = 1/abs(np.imag(Kr+4/3*Gmm)/np.real(Kr+4/3*Gmm))
        Qs = 1/abs(np.imag(Gmm)/np.real(Gmm))
        return Qp, Qs
    
    @staticmethod 
    def dvorkin_mavko (dry_bulk, 
                       dry_shear, 
                       mineralset_bulk, 
                       water_bulk, 
                       hydrocarbon_bulk, 
                       porosity, 
                       water_saturation, 
                       irreducible_water_saturation):
        """
        Dvorkin and Mavko (2006) model of P-wave quality factor for a partial water saturated media.
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        water_bulk : np.ndarray
            Effective bulk modulus of the water (GPa)
        porosity : np.ndarray
            Porosity
        hydrocarbon_bulk : np.ndarray
            Effective bulk modulus of the hydrocarbon (GPa)
        Porosity: np.ndarray
            Porosity
        water_saturation: np.ndarray
            Water saturation (V/V)
        irreducible_water_saturation : np.ndarray
            Irreducible water saturation (V/V)
            
        Returns
        -------
        p_qfactor : np.ndarray
            Frequency-dependant P-wave Quality Factor of saturated rock (GPa)
            
        Notes
        -----
        - Important: In case if only P-wave velocity/modulus is avaialble one can use the Vp-only 
          fluid substitution model which is an approximate form of this model. In order to implement 
          this model simply set the Gdry=0 and replace all input bulk moduli (for solids) with their
          corresponding P-wave modulus. i.e.
          >>> Qp = QualityFactorMethod.dvorkin_mavko(Mdry, 0, Mm, Kw, Khc, phi, Sw, Swir)
        - To Estimate Qs, use `qp2qs` function.
        - It is assumed that the difference between M0 and Minf is nonzero only at water saturation 
          larger than irreducible water saturation Swirr. For SW <= Sirr, Minf = M0, i.e., 1/Qp = 0.
        - The difference between low- and high-frequency limit velocity estimates may give rise to 
          noticeable P-wave attenuation if elastic heterogeneity in rock is substantial. Therefore, 
          The necessary condition for attenuation is elastic heterogeneity in rock. This heterogenity,
          among many other reasons, may either be due to partial water saturation ('dvorkinmavko') or
          layering in media (`QualityFactorMethod.dvorkin_mavko_layered`).
        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Dvorkin, J., Mavko, G., and Nur, N., 1995, Squirt flow in fully saturated rocks. GEOPHYSICS 60: 97-107

        """
        fluidset_bulk = 1/(  water_saturation/water_bulk + (1-water_saturation)/hydrocarbon_bulk )
        K0 =  mineralset_bulk*  ( porosity*dry_bulk - (1+porosity)*fluidset_bulk*dry_bulk/mineralset_bulk + fluidset_bulk ) \
                / ( (1-porosity)*fluidset_bulk + porosity*mineralset_bulk - fluidset_bulk*dry_bulk/mineralset_bulk )
        M0 = K0 + 4/3*dry_shear

        Kflir = 1/(  irreducible_water_saturation/water_bulk + (1-irreducible_water_saturation)/hydrocarbon_bulk )
        KP =  mineralset_bulk*  ( porosity*dry_bulk - (1+porosity)*water_bulk*dry_bulk/mineralset_bulk + water_bulk ) \
                / ( (1-porosity)*water_bulk + porosity*mineralset_bulk - water_bulk*dry_bulk/mineralset_bulk )
        Kmir =  mineralset_bulk*  ( porosity*dry_bulk - (1+porosity)*Kflir*dry_bulk/mineralset_bulk + Kflir ) \
                / ( (1-porosity)*Kflir + porosity*mineralset_bulk - Kflir*dry_bulk/mineralset_bulk )

        Minf = 1/( (water_saturation-irreducible_water_saturation)/(1-irreducible_water_saturation)/(KP+4/3*dry_shear) + (1-water_saturation)/(1-irreducible_water_saturation)/(Kmir+4/3*dry_shear) )

        Qp = 2/((Minf-M0)/np.sqrt(Minf*M0))
        return Qp

    @staticmethod 
    def dvorkin_mavko_layered (dry_bulk, 
                               dry_shear, 
                               mineralset_bulk, 
                               mineralset_shear, 
                               water_bulk, 
                               porosity, 
                               window_length):
        """
        P-wave quality factor for a fully water saturated layered media. 
                
        Parameters
        ----------
        dry_bulk : np.ndarray
            Effective bulk modulus of the rock frame (GPa)
        dry_shear : np.ndarray
            Effective shear modulus of the rock frame (GPa)
        mineralset_bulk : np.ndarray
            Bulk modulus of the mineral (GPa)
        mineralset_shear : np.ndarray
            Shear modulus of the mineral (GPa)
        water_bulk : np.ndarray
            Effective bulk modulus of the fluid (GPa)
        porosity : np.ndarray
            Porosity
        window_length : np.ndarray
            Window length of moving average (samples)
            
        Returns
        -------
        p_qfactor : np.ndarray
            Frequency-dependant P-wave Quality Factor of saturated rock (GPa)
            
        Notes
        -----
        - The difference between low- and high-frequency limit velocity estimates may give rise to
          noticeable P-wave attenuation if elastic heterogeneity in rock is substantial. Therefore,
          The necessary condition for attenuation is elastic heterogeneity in rock. This heterogenity,
          among many other reasons, may either be due to partial water saturation or layering in media.
        - To Estimate Qs, use `qp2qs` function.

        
        References
        ----------
        .. [1] Mavko, G., Mukerji, T. and Dvorkin, J., 2020. The rock physics handbook. Cambridge university press.
        .. [2] Dvorkin, J., Mavko, G., and Nur, N., 1995, Squirt flow in fully saturated rocks. GEOPHYSICS 60: 97-107

        """
        Mdry = dry_bulk + 4/3*dry_shear
        Mm = mineralset_bulk + 4/3*mineralset_shear
        movingmean_filter = np.ones(window_length)/window_length
        phiavg = np.convolve( porosity, movingmean_filter, mode='same')
        Mdryavg = 1 / np.convolve(1/Mdry, movingmean_filter, mode='same')
        Mmavg = 1/ np.convolve(1/Mm,movingmean_filter, mode='same')

        M0 =  Mmavg*  ( phiavg*Mdryavg - (1+phiavg)*water_bulk*Mdryavg/Mmavg + water_bulk ) \
                / ( (1-phiavg)*water_bulk + phiavg*Mmavg - water_bulk*Mdryavg/Mmavg )

        Minf =  Mm*  ( porosity*Mdry - (1+porosity)*water_bulk*Mdry/Mm + water_bulk ) \
                / ( (1-porosity)*water_bulk + porosity*Mm - water_bulk*Mdry/Mm )
        Minf = 1/ np.convolve(1/Minf, movingmean_filter, mode='same')
        p_qfactor = 2/((Minf-M0)/np.sqrt(Minf*M0))
        return p_qfactor
