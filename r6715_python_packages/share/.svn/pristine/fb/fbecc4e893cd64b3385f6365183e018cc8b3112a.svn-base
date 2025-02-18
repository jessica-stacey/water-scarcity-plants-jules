# -*- coding: iso-8859-1 -*-
import os
import math

import numpy as np

import soil_cosby_parameters


'''
Functional forms of soil moisture stress parameters, from a variety
of published papers and computer codes. 

This is a work-in-progress and could contain many errors: please get in touch
if you want to help improve this script or notice any bugs.

Karina Williams


Notes

The equations from Huntingford et al 2015 (including the Tardieu model) are implemented 
in a separate script: huntingford2015_equations.py

"At a potential of -33 kPa, or 1/3 bar, (-10 kPa for sand), soil is at field capacity... At 
a potential of -1500 kPa, soil is at its permanent wilting point, meaning that soil water 
is held by solid particles as a 'water film that is retained too tightly to be taken up by 
plants". (wikipedia water potential page)  

Good description: http://nrcca.cals.cornell.edu/soil/CA2/CA0212.1-3.php  
'''


def jules_soil_factor(soil_water_percent, soil, fsmc_shape=0, **kwargs):

    if fsmc_shape == 0:
        beta = jules_soil_factor_fsmc_shape_0(soil_water_percent, soil, **kwargs)
    elif fsmc_shape == 1:
        beta = jules_soil_factor_fsmc_shape_1(soil_water_percent, soil, **kwargs)
    else:
        raise ValueError('this value of fsmc_shape has not yet been implemented')
    return beta


def jules_soil_factor_fsmc_shape_0(soil_water_percent, soil, p0=None):
    '''
    JULES soil water stress factor for fsmc_shape=0.
    (see 
    https://code.metoffice.gov.uk/trac/jules/attachment/wiki/PegSoilMoistureStressVegMeeting2/soil_moisture_stress_jules.pdf
    for a full description)
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * p0: pft-dependent parameter
    '''

    if p0 is None:
        p0 = 0.0

    v_sat = soil.jules_soil_parameters['sm_sat']
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']

    sthu = (soil_water_percent / 100.0) / v_sat

    if abs(v_crit - v_wilt) > 0.0:
        beta = ( sthu * v_sat - v_wilt ) \
               / ( v_crit - v_wilt) \
               / ( 1.0 - p0 )
    else:
        beta = 0.0 * sthu 

    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta


def jules_soil_factor_fsmc_shape_1(soil_water_percent, soil, psi_open=0.0, psi_close=-1.5E6):
    '''
    JULES soil water stress factor for fsmc_shape=1.
    (see 
    https://code.metoffice.gov.uk/trac/jules/wiki/ticket/541/TicketDetails
    for a full description)
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * psi_open: Potential above which the soil moisture stress factor fsmc is one. 
       Unit: Pa. Allowed range: must be negative. 
     * psi_closed: Potential below which the soil moisture stress factor fsmc is zero. 
       Unit: Pa. Allowed range: must be negative.   
    '''
    
    if psi_open > 0.0:
        raise ValueError('psi_open should be <= 0.0')

    if psi_close > 0.0:
        raise ValueError('psi_close should be <= 0.0')

    rho_w = 999.97  # density of water kg m-3 (at 4 degC)
    g = 9.81  # acceleration due to gravity in m s-2

    psi = - ( soil.jules_soil_parameters['sathh'] * rho_w * g ) * \
         (soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']) \
         ** ( -soil.jules_soil_parameters['b'] )
    
    if abs(psi_close - psi_open) > 0.0:
        beta = ( psi_close - psi ) \
               / (  psi_close - psi_open )
    else:
        beta = 0.0 * psi

    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta
    

def gleamv3_at_vodmax(soil_water_percent, soil):
    '''
    Ref: Section 2.2.3 in
    GLEAM v3: satellite-based land evaporation and root-zone soil moisture, 
    Martens et al 2017
    http://www.geosci-model-dev.net/10/1903/2017/
    
    Args:
     * soil_water_percent: soil water as a percentage (of wettest layer)
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
    
    "Multiplicative stress factor S ranging between 0 (maximum stress
and thus no evaporation) and 1 (no stress and thus potential
evaporation) is defined... However, based on experimental evidence, a non-
linear response of S to soil moisture is expected for most
vegetation types (e.g. Colello et al., 1998; Serraj et al., 1999;
Ronda et al., 2002; Combe et al., 2016). As a consequence,
a non-linear stress function for both tall and short vegetation
is re-introduced in GLEAM v3:

S=sqrt(VOD/VODmax)(1-(\frac{w_C-w^{(w)}}{w_c-w_wp})^2)
w^{(w)} is the soil moisture content of the wettest layer, assuming that
plants withdraw water from the layer in which it is more easily accessible.

    Doesn't define 'critical' soi moisture, so am assuming this is
    field capacity.
    '''
    
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']
    vol_soil_water = soil_water_percent / 100.0
    # assuming VOD=VODmax
    beta = 1.0 - ( (v_crit - vol_soil_water) / (v_crit - v_wilt) ) ** 2.0 
    beta[vol_soil_water <= v_wilt] = 0.0 
    beta[vol_soil_water >= v_crit] = 1.0 
    
    return beta


def maizegro(soil_water_percent, soil):
    '''
    Ref: Section 2.1 in 
    http://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=3680&context=etd
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
    
    Notes:
    
    Also see maizegro code
    https://r-forge.r-project.org/scm/viewvc.php/pkg/R/waterStress.R?view=markup&root=biocro
    which has this and other, simpler functions
    
    Maizegro has a look up table for filed capacity and wilting point, which are not the
    same as the ones calculated in soil_cosby_parameters.
    E.g. for silty clay loam, MaizeGro has  
    v_crit = 0.37
    v_wilt = 0.21
    See https://r-forge.r-project.org/scm/viewvc.php/pkg/R/showSoilType.R?view=markup&revision=73&root=biocro
    '''
    
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']
    
    aw = soil_water_percent / 100.0
    aw = np.maximum(aw, 0.0)   
    
    slp = ( 1.0 - v_wilt ) / ( v_crit - v_wilt )
           
    intept = 1.0 - v_crit * slp
    
    theta = slp * aw + intept
    
    x = -2.5 # in MaizeGro code, functions wtrstr and wsRcoef look like they have two different values for this -1.0, -2.5)
    
    beta = (1.0 - np.exp(x * (theta - v_wilt) / (1.0 - v_wilt))) / (
            1.0 - np.exp(x))
    
    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)
    
    return beta


def SiB3_SR(soil_water_percent, soil):
    '''
    Ref: Section 3.1 in
    http://biocycle.atmos.colostate.edu/Papers_files/Baker.Amazon.roots.pdf
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    Notes:    
     * Applies to whole column (3.5m for SiB-SR, 10m for SiB3-DS)
     * They also try an option that uses hydraulic re-distribution.
     * Wilting point is defined as a moisture potential of -150 m.    
     * In SiB3crop, it looks like the same expression is used (begtem.F90)
    '''
    
    wssp = 0.2
    
    wcolumn = soil_water_percent/100.0 - soil.jules_soil_parameters['sm_wilt']
    wcolumn = np.maximum(wcolumn, 0.0)   
    
    wmax = soil.jules_soil_parameters['sm_crit'] - soil.jules_soil_parameters['sm_wilt']
    
    wcolumn_by_wmax = wcolumn / wmax
    
    beta = (1.0 + wssp) * wcolumn_by_wmax / (wssp + wcolumn_by_wmax)    
    
    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)
    
    return beta


def ibis_2p6b4(soil_water_percent, soil):
    ''' 
    Ref: Subroutine drystress from physiology.f
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    Code extracts:
    
    c stressfac determines the shape of the soil moisture response
      stressfac = -5.0
      
      znorm = 1.0 - exp(stressfac)
      
     c plant available water content (fraction)
          awc = min (1.0, max (0.0,
     >              (wsoi(i,k)*(1 - wisoi(i,k))   - swilt(i,k)) /
     >              (sfield(i,k) - swilt(i,k))
     >              )         )
     
          zwilt = (1. - exp(stressfac * awc)) / znorm
          
    uses variables (comsoi.h):
     >  wsoi(npoi,nsoilay),      ! fraction of soil pore space containing liquid water
     >  wisoi(npoi,nsoilay),     ! fraction of soil pore space containing ice
     >  sfield(npoi,nsoilay),    ! field capacity soil moisture value (fraction of pore space)
     >  swilt(npoi,nsoilay),     ! wilting soil moisture value (fraction of pore space)
     
    zwilt is then weighted using froot to get
     >  stressl(npoi,nsoilay),   ! soil moisture stress factor for the lower canopy (dimensionless)
     >  stressu(npoi,nsoilay),   ! soil moisture stress factor for the upper canopy (dimensionless)
     
    Here, assume wisoi = 0 and there's a constant factor relating 
    soil water-filled pore space (%)  
    and vol soil moisture
    (e.g. http://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs142p2_053260.pdf)
    e.g. could assume vol_rock_etc = 0.5*(vol_tot) where vol_tot = v_air + v_water + v_rock_etc
    
    '''  
    stressfac = -5.0
    znorm = 1.0 - np.exp(stressfac)
    
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']
    
    awc = (soil_water_percent/100.0 - v_wilt) / (v_crit - v_wilt)
    awc = np.maximum(0.0, awc)
    awc = np.minimum(1.0, awc)
    
    beta = (1.0 - np.exp(stressfac * awc)) / znorm

    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)
    
    return beta


def fao_evapotranspiration(soil_water_percent, soil, ETc_in_mm_per_day=None, p_Table22=0.55):
    '''
    Ref: http://www.fao.org/docrep/X0490E/x0490e0e.htm
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * p_Table22: depletion fraction for ET >> 5 mm/day, as given in table 22. 
       For "Maize, Field (grain) (field corn)", p_Table22 is 0.55 (there's lots of other crops too).
    
    Notes:
    
    Amount of water that is available for evaporation is affected by crop type.

    Eq 82
     TAW = 1000(qFC - qWP) Zr
    Eq 83
     RAW = p TAW    

    TAW the total available soil water in the root zone [mm],
    qFC the water content at field capacity [m3 m-3],
    qWP the water content at wilting point [m3 m-3],
    Zr the rooting depth [m]. (max Zr is given in table 22. Is 1.0-1.7 for "Maize, Field (grain) (field corn)")
    RAW the readily available soil water in the root zone [mm],
    p average fraction of Total Available Soil Water (TAW) that can be depleted 
       from the root zone before moisture stress (reduction in ET) occurs [0-1].
    p = p_Table22 + 0.04 (5 - ETc) where the adjusted p is limited to 0.1 < p < 0.8 and ETc is in mm/day. 
    
    "Field capacity is the amount of water that a well-drained soil should hold against gravitational forces, 
    or the amount of water remaining when downward drainage has markedly decreased... Wilting point is the 
    water content at which plants will permanently wilt."

    It also says that
    "To express the tolerance of crops to water stress as a function of the fraction (p) of TAW is not 
    wholly correct. The rate of root water uptake is in fact influenced more directly by the potential energy
    level of the soil water (soil matric potential and the associated hydraulic conductivity) than by 
    water content. As a certain soil matric potential corresponds in different soil types with different 
    soil water contents, the value for p is also a function of the soil type. Generally, it can be stated 
    that for fine textured soils (clay) the p values listed in Table 22 can be reduced by 5-10%, while for 
    more coarse textured soils (sand), they can be increased by 5-10%."
    
    Also, table 19 has WP and FC for different types of soil e.g.
     Silt clay loam FC 0.30 - 0.37, WP 0.17 - 0.24
    Recall our parameters for silty clay loam (calculated by soil_cosby_parameters) were crit 0.3917, wilt 0.2488
     i.e. bit higher
     
    '''

    p = p_Table22 + 0.04 * (5.0 - ETc_in_mm_per_day)

    p = np.maximum(p, 0.1)
    p = np.minimum(p, 0.8)

    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']

    beta = (soil_water_percent / 100.0  - v_wilt) / (v_crit - v_wilt) / (1.0 - p)

    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta


def oleson_soil_factor(soil_water_percent, soil, psi_open=-74.0, psi_closed=-275.0):
    '''
    Ref: From Oleson et al 2008 (http://onlinelibrary.wiley.com/doi/10.1029/2007JG000563/abstract)
    but simplified by assuming no frozen water in soil and temperature is above a threshold.
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * psi_open: soil water potential at stomata fully open in m
       For C3 grass, C4 grass and two crops, psi_open = -74.0m (table E1)
     * psi_closed: soil water potential at full stomatal closure in m
       For C3 grass, C4 grass and two crops, psi_closed = -275.0m (table E1)
    
    In CLM, this factor is calculated for each soil layer and then weighted according to the 
    fraction of roots in that layer (the root distribution is the sum of two exponential 
    decays (CLM 4 tech note):
        exp(-r_a*z) + exp(-r_b*z), r_a=6, r_b=3 for crops). It directly multiplies vcmax.    
    '''
    
    psi = - ( soil.jules_soil_parameters['sathh'] ) * \
         (soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']) \
         ** ( -soil.jules_soil_parameters['b'] )
    
    if abs(psi_closed - psi_open) > 0.0:
        beta = ( psi_closed - psi ) \
               / (  psi_closed - psi_open )
    else:
        beta = 0.0 * psi

    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta


def de_Vries_soil_factor(soil_water_percent, soil, lai=None, trc_in_mm_per_day=None, 
    sensitivity_to_drought_stress=0.65, sensitivity_to_flooding=0.6):    
    '''
    Ref: Simulation of Ecophysiological Processes of Growth in Several Annual Crops,
    F. W. T. Penning de Vries, 1989, http://books.google.co.uk/books?id=G-cjxJlr71wC
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * lai: leaf area index
     * trc_in_mm_per_day: potential canopy transpiration
     * sensitivity_to_drought_stress: See section 4.2.2. 0.2 is not resistant, 1 is resistant.
       sensitivity_to_drought_stress=0.65 for cereals (table 24).
     * sensitivity_to_flooding: See section 4.2.2. 0.2 is sensitive to flooding, 1.0 is insensitive
       sensitivity_to_flooding=0.6 for cereals (table 24).
    
    Penning de Vries applies a constant ratio of transpiration to gross photosynthesis
    transpiration is limited by water availability, 
    Says that water uptake by each layer is not determined by the density of roots at each layer.
    Instead, assumes the water uptake per depth of roots is the same down to a lower limit.
    
    Extracts from the code in the appendix:
    
    p263 is FUWS function
    WCL < WCFC
    FUWSX = (WCL - WCWP) /(WCX-WCWP)
    else
    FUWSX = 1-(1-WFSC)*(WCL-WCFC)/(WCST-WCFC)

    WFSC flooding sens
    WCL water content
    WCST: at sat
    WCWP: wilt
    WCFC: field cap
    WCX = WCWP + (WCFC-WCWP)*(1.0-MIN(1,MAX(0,SDPF))
    WSSC=water stress sensitivity coeff (0.5 for rice)

    if WSSC >= 0.6:
    SDPF = 1.0 /(A+B*ALVMAX*TRC/(ALV+1.0E-10))-(1-WSSC)*0.4

    DATA A,B,ALVMAX/0.76,0.15.2./

    ALV: leaf area
    TRC: potential canopy transpiration 
    TRW: canopy transpiration with water stress

    '''

    field_capacity = -0.1 # bar
    field_capacity *= 1.0E5 #Pa 
    wilting_point = -16.0 #bar 
    wilting_point *= 1.0E5 #Pa 

    v_sat = soil.jules_soil_parameters['sm_sat']
    v_fc = soil._theta(psi=abs(field_capacity))
    v_wilt = soil._theta(psi=abs(wilting_point))
     
    # these are hardwired in
    lai_max = 2.0 
    a = 0.76
    b = 0.15

    if sensitivity_to_drought_stress < 0.6:
        raise Exception('have not implemented this case')
    sdpf = 1.0 / (a + b * lai_max * trc_in_mm_per_day / (lai + 1.0E-10))-(1.0-sensitivity_to_drought_stress)*0.4
    sdpf = min(1.0, max(0.0, sdpf))

    v_x = v_wilt + (v_fc - v_wilt) * (1.0 - sdpf)

    soil_water_frac = soil_water_percent / 100.0

    beta = 0.0 * soil_water_frac 

    for i,x in enumerate(soil_water_frac):
        if x < v_fc:        
            beta[i] = (x - v_wilt) \
                   / (v_x - v_wilt + 1.0E-10)
        else:
            beta[i] = 1.0 - (1.0 - sensitivity_to_flooding) * (x - v_fc) / (v_sat - v_fc + 1.0E-10)
 
    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta


def zhou_rt_factor(soil_water_percent, soil, psi_f=-0.99, s_f=11.97,
                   g1star=4.72, vpd_in_kPa=2.0, b=1.61):
    '''
    relative transpiration when plant is vcmax limited
    Ref: Zhou et al 2013
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
    
    Kwargs:
     * psi_f: the water potential at which psi_pd decreases to half its max value
       For medit. herbs, psi_f = -0.99 (Zhou et al 2013)
     * s_f: sensitivity parameter 
       For medit. herbs, s_f = 11.97 (Zhou et al 2013)
     * g1star: slope of the sensitivity of gs to A at psi_pd=-0.5MPa
       For medit. herbs, g1star = 4.72 (Zhou et al 2013).
       In (kPa)^{-0.5}, I think, same as g1.
     * vpd_in_kPa: vapor pressure deficit at the leaf surface
     * b: fitted parameter multiplying psi_pd in exponent in g1
       For medit. herbs, b = 1.61 (Zhou et al 2013). 
       In MPa-1, I think (I checked a malacophyll angiosperm tree against fig 4
       at psi_pd = -1MPa and it looked about right) 
       
     g0 (gs when A is zero in mol m-2 s-1) is assumed to be zero.
       
    '''    

    f_vcmax = kauwe_vcmax_factor(soil_water_percent, soil, psi_f=psi_f, s_f=s_f)
    
    rho_w = 999.97  # density of water kg m-3 (at 4 degC)
    g = 9.81  # acceleration due to gravity in m s-2    
        
    # pre-dawn water leaf potential
    psi_pd = (rho_w * g) * ( - soil.jules_soil_parameters['sathh'] ) * (
            soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']
            ) ** ( -soil.jules_soil_parameters['b'] )
    psi_pd /= 1.0E6 #MPa
            
    g1wet = g1star / np.exp(- b * (-0.5))        
            
    g1 = g1wet * np.exp( b * psi_pd)

    beta = ( 1.0 + g1 / np.sqrt(vpd_in_kPa)) \
             /  ( 1.0 + g1wet / np.sqrt(vpd_in_kPa)) * f_vcmax 

    return beta


def kauwe_vcmax_factor(soil_water_percent, soil, psi_f=-0.99, s_f=11.97):
    '''
    Ref: http://www.biogeosciences.net/12/7503/2015/bg-12-7503-2015.html
    (Adds to CABLE)
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
    
    Kwargs:
     * psi_f: the water potential at which psi_pd decreases to half its max value
       For medit. herbs, psi_f = -0.99 (Zhou et al 2013)
     * s_f: sensitivity parameter 
       For medit. herbs, s_f = 11.97 (Zhou et al 2013)
    '''    

    rho_w = 999.97  # density of water kg m-3 (at 4 degC)
    g = 9.81  # acceleration due to gravity in m s-2
        
    # pre-dawn water leaf potential
    # I'm not sure I've applied this properly - check it!
    psi_pd = (rho_w * g) * ( - soil.jules_soil_parameters['sathh'] ) * (
            soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']
            ) ** ( -soil.jules_soil_parameters['b'] )
                  
    psi_pd /= 1.0E6
              
    beta = (1.0 + np.exp(s_f * psi_f)) / (1.0 + np.exp(s_f*(psi_f - psi_pd)))
    
    return beta
    
    
def wofost(soil_water_percent, soil, et0_in_cm_per_day=None, No_cg=4.5):
    '''
    Ref: Section 6 (Soil water balance) from http://supit.net/
 
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    Kwargs:
    * et0_in_cm_per_day is potential evapotranspiration rate in cm d-1  (for a closed canopy)  
      NIn the code, the calculated ETO gets modified by a crop-specific correction, but this is 1 for grain maize:
      https://github.com/ajwdewit/WOFOST/blob/5dfc3521d4f1c034a022ad7cf4ac13a2feb823fc/sources/w60lib/evtra.for#L46
    * No_cg is crop group number (=1 to 5, Doorenbos et al., 1978)
      For grain maize, No_cg = 4.5, see
      https://github.com/ajwdewit/WOFOST/blob/5dfc3521d4f1c034a022ad7cf4ac13a2feb823fc/cropd/grain_maize.crp#L120
    '''
 
    # alpha_p is a dimensionless regression constant (=0.76 van Diepen et al., 1988)
    alpha_p = 0.76
  
    # beta_p is a regression constant (=1.5 van Diepen et al., 1988), units d cm-1
    beta_p = 1.5
 
    if No_cg < 3.0:
        raise Exception('this needs a different form of p which I have not implemented')
 
    # p is fraction of easily available soil water 
    # eqn 6.10:
    p = 1.0 / (alpha_p + beta_p * et0_in_cm_per_day) - 0.1 * (5.0 - No_cg)
     
    # p must be between 0.1 and 0.95
    #https://github.com/ajwdewit/WOFOST/blob/5dfc3521d4f1c034a022ad7cf4ac13a2feb823fc/sources/w60lib/sweaf.for#L49
    p = np.maximum(p, 0.1)
    p = np.minimum(p, 0.95)
 
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_fc = soil.jules_soil_parameters['sm_crit']
      
    v_crit = (1.0 - p)*(v_fc - v_wilt) + v_wilt
 
    # beta is "Reduction factor for transpiration in case of water shortage"
    beta = (soil_water_percent / 100.0  - v_wilt) / (v_crit - v_wilt) 
 
    # beta is between 0 and 1:
    #https://github.com/ajwdewit/WOFOST/blob/5dfc3521d4f1c034a022ad7cf4ac13a2feb823fc/sources/w60lib/evtra.for#L67
    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)
 
    return beta
 

def sinclair_soil_factor(soil_water_percent, soil, psi_e_in_Pa=None):
    '''
    Ref: 
    Theoretical Analysis of Soil and Plant Traits Influencing Daily Plant Water Flux on Drying Soils
    Sinclair 2005
    https://dl.sciencesocieties.org/publications/aj/abstracts/97/4/1148
   
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    Kwargs:
     * psi_e_in_Pa is the average/typical water potential of the bulk leaf epidermis (constant)
    '''

    psi = - ( soil.jules_soil_parameters['sathh'] ) * \
         (soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']) \
         ** ( -soil.jules_soil_parameters['b'] ) # in m
    
    rho_w = 999.97  # density of water kg m-3 (at 4 degC)
    g = 9.81  # acceleration due to gravity in m s-2

    # convert from m to Pa
    psi *= (rho_w * g)
   
    beta = 1.0 - psi / psi_e_in_Pa
    beta = np.maximum(beta, 0.0)
    beta = np.minimum(beta, 1.0)

    return beta
    
 
def kim1991_fife_soil_factor(soil_water_percent, soil, grass_species='Andropogon gerardii'):
    '''
    Ref: 
    Modeling canopy stomatal conductance in a temperate grassland ecosystem
    Joon Kim, Shashi B. Verma
    http://dx.doi.org/10.1016/0168-1923(91)90028-o
   
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    Kwargs:
     * grass_species: grass species used in study.
    
    Data fit to soil moisture data in the primary root zone (0-1.4).
    
    There's also a model against leaf water potential (piecewise linear, threshold -1MPa.)
    but not sure how to interpret this, as water potential is from hourly measurements,
    but there's a separate term for VPD dependence.
    
    The site was 27.1% Andropogon gerardii, 22.2% Sorghastrum nutans, 
    16.6% Panicum vergatum.
    '''

    if grass_species not in ['Andropogon gerardii', 'Sorghastrum nutans', 'Panicum vergatum']:
        raise UserWarning('grass species not recognised')

    if grass_species == 'Andropogon gerardii':
        # Big bluestem, C4
        a4 = 0.029 # +/- 0.008 
    elif grass_species == 'Sorghastrum nutans':
        # Indiangrass, C4
        a4 = 0.013 # +/- 0.005
    elif grass_species == 'Panicum vergatum':
        # Switchgrass, C4
        a4 = 0.01 # +/- 0.006
    else:
        raise UserWarning('grass species not recognised')

    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_crit = soil.jules_soil_parameters['sm_crit']
    
    awc = (soil_water_percent/100.0 - v_wilt) / (v_crit - v_wilt)
    awc = np.maximum(0.0, awc)
    awc = np.minimum(1.0, awc)
    
    aw_percent = awc * 100.0
    
    beta = 1.0 - np.exp( - a4 * aw_percent )
    # normalise
    beta /= 1.0 - np.exp( - a4 * 100.0 ) 
    

    return beta


def cable_haverd(soil_water_percent, soil, gamma=0.03):
    '''
    Ref: Haverd et al 2016
    https://www.geosci-model-dev.net/9/3111/2016/gmd-9-3111-2016.pdf
    New function for CABLE r3432 based on Lai and Katul 2000 with
    gamma tuned to flux LE.
    Wettest layer gets used.
    
    02/10/2017: edited based on email from Martin De Kauwe,
    which says
    
    beta = 0.0
    x = gamma / max(1.0e-3, theta - wp)
    frwater = max(1.0e-4,((theta - wp)/sat)**x)
    beta = min(1.0, max(beta, frwater))
    
    Args:
     * soil_water_percent: soil water as a percentage
     * soil: soil properties object (class soil_cosby_parameters.SoilType) 
     
    KWargs:
     * gamma: free parameter. Defaults to 0.03, which gave the best fit in
       Haverd et al 2016.
     
    '''
    
    v_wilt = soil.jules_soil_parameters['sm_wilt']
    v_sat = soil.jules_soil_parameters['sm_sat']
    v = soil_water_percent / 100.0
    
    x = gamma / np.maximum(v  - v_wilt, 1.0E-3)
    beta = ( (v  - v_wilt) / v_sat ) ** x

    beta = np.maximum(beta, 1.0E-4)
    beta = np.minimum(beta, 1.0)

    return beta

def calc_root_frac_exponential(rootd=None, dz=None):
    return calc_root_frac_exponential(rootd=None, dz=None, fsmc_mod=0)
            

def calc_root_frac(rootd=None, dz=None, fsmc_mod=None):
    '''
    Calculates the weighting for each soil layer using the root fraction
    Copied from root_frac_jls.F90 
    Keeping the syntax as close as possible to root_frac_jls.F90
    for easy comparison.
    
    KWargs:
     * rootd: pft-dependent e-folding depth in the root distribution svn add beta_peg/beta_function_for_Rob.py
       (JULES_PFTPARM namelist)
     * dz: numpy array containing thicknesses of the soil layers. Often set to np.array([0.1, 0.25, 0.65, 2.0]).
    '''
    if fsmc_mod not in [0,1]:    
        raise Exception('this fsmc_mod is not implemented')

    z2 = 0.0
    ztot = 0.0
    ftot = 0.0
    p = 1.0

    nshyd = len(dz)
    f_root = np.zeros(nshyd)

    if fsmc_mod == 1:
        for n in range(0, nshyd):
            z1 = z2
            z2 = z2 + dz[n]
            ztot = ztot + dz[n]
  	    if z1 > rootd: 
  	      f_root[n] = 0.0
  	    elif z2 < rootd:
  	      f_root[n] = dz[n]
  	    else:
  	      f_root[n] = rootd - z1

        ftot = np.minimum(rootd, ztot)

        for n in range(0, nshyd):
            f_root[n] = f_root[n] / ftot

    elif fsmc_mod == 0: 
        for n in range(0, nshyd):
            z1 = z2
            z2 = z2 + dz[n]
            ztot = ztot + dz[n]
            f_root[n] = math.exp( - p * z1 / rootd) - math.exp( - p * z2 / rootd)
                
        ftot = 1.0 - math.exp( - p * ztot / rootd)
        for n in range(0, nshyd):
            f_root[n] = f_root[n] / ftot
    else:
        raise Exception('this fsmc_mod is not implemented')
      
    if not np.allclose(np.sum(f_root), 1.0):
        raise UserWarning('something has gone wrong: f_root does not add up to 1')    

    return f_root
