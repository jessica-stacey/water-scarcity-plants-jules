# -*- coding: iso-8859-1 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
Functions to explore JULES soil hydrology, for use in the sm-stress JPEG

This is a work-in-progress and could contain many errors: please get in touch
if you want to help improve this script or notice any bugs.

Karina Williams
'''


# from water_constants_mod.F90@vn4.8
RHO_WATER = 1000.0  # density of pure water (kg/m3)

# from jules_soil_mod.F90@vn4.8
GAMMA_W = 1.0  # Forward timestep weighting


def hydraulic_conductivity(b, ks, thetak):
    '''
    Following hyd_con_ch_mod.F90@vn4.8

    Args
                                                         
     * b (float): Exponent in conductivity and soil water suction fits.
     * ks (float): The saturated hydraulic conductivity (kg/m2/s)
     * thetak (float): Fractional saturation.

    Returns

     * k (float): The hydraulic conductivity (kg/m2/s).
     * dk_dthk (float): The rate of change of K with THETAK (kg/m2/s).

    '''
    small_value = 1.0E-7 # made something up here. In fortran, was
                        # small_value=EPSILON(0.0)

    dk_dthk=0.0
    if (thetak >= small_value) and (thetak < 1.0):
        k=ks*thetak**(2.0*b+3.0)
        dk_dthk=(2.0*b+3.0)*ks*(thetak**(2.0*b+2.0))
    elif thetak < small_value:
        k=0.0
    else:
        k=ks

    return k, dk_dthk


def gauss(a,b,c,d,xmin,xmax):
    '''
    Based on gauss_jls_mod.F90@vn4.8  
    Note that in the fortran the arrays are 1...nlevs whereas here they are 0...nlevs-1.

    Args
                                                     &
    * a(nlevs): Matrix elements corresponding to the coefficients of X(n-1).
    * b(nlevs): Matrix elements corresponding to the coefficients of X(n).
    * c(nlevs): Matrix elements corresponding to the coefficients of X(n+1).
    * d(nlevs): Matrix elements corresponding to the RHS of the equation.
    * xmin: Minimum permitted value of X.
    * xmax: Maximum permitted value of X.

    returns
    * x(nlevs): solution
    '''

    nlevs = len(a)
    
    # initialise (not in fortran code)
    adash = np.zeros((nlevs,))
    bdash = np.zeros((nlevs,))
    ddash = np.zeros((nlevs,))

    x = d.copy()

    #-----------------------------------------------------------------------
    # Upward Sweep: eliminate "C" elements by replacing nth equation with:
    #                  B'(n+1)*Eq(n)-C(n)*Eq'(n+1)
    # where "'" denotes a previously tranformed equation. The resulting
    # equations take the form:
    #                A'(n) X(n-1) + B'(n) X(n) = D'(n)
    # (NB. The bottom boundary condition implies that the NLEV equation does
    #  not need transforming.)
    #-----------------------------------------------------------------------

    adash[nlevs-1]=a[nlevs-1]
    bdash[nlevs-1]=b[nlevs-1]
    ddash[nlevs-1]=d[nlevs-1]

    for n in range(nlevs-2, -1, -1):  # nlevs-2,...,0
        adash[n]=bdash[n+1]*a[n]
        bdash[n]=bdash[n+1]*b[n]-c[n]*adash[n+1]
        ddash[n]=bdash[n+1]*d[n]-c[n]*ddash[n+1]

    #-----------------------------------------------------------------------
    # Top boundary condition: A(1) = 0.0 , allows X(1) to be diagnosed
    #-----------------------------------------------------------------------

    if bdash[0] != 0.0: 
        x[0]=ddash[0]/bdash[0]
    else:
        raise UserWarning('divide by zero warnings at top layer')

    x[0] = max(x[0],xmin[0])
    x[0] = min(x[0],xmax[0])

    #-----------------------------------------------------------------------
    # Downward Sweep: calculate X(n) from X(n-1):
    #                X(n) = (D'(n) - A'(n) X(n-1)) / B'(n)
    #-----------------------------------------------------------------------

    for n in range(1, nlevs):
        if bdash[n] != 0.0:
          x[n]=(ddash[n]-adash[n]*x[n-1])/bdash[n]
        else:
          raise UserWarning('divide by zero warnings')

        x[n] = max(x[n], xmin[n])
        x[n] = min(x[n], xmax[n])

    return x


def darcy(b, dz1, dz2, ks, sathh, sthu1, sthu2, l_dpsids_dsdz=False):
    '''
    Following darcy_ic_mod.F90@vn4.8 and darcy_ch_mod.F90@vn4.8
    Note that in the fortran the elements of b, sathh and theta are 1,2
    whereas here they are 0,1.

    Args
                                                       
    * b(2): Clapp-Hornberger exponent (zeroth element upper layer, first element lower layer).
    * dz1: Thickness of the upper layer (m).
    * dz2: Thickness of the lower layer (m).
    * ks: Saturated hydraulic conductivity (kg/m2/s). One number for both layers
    * sathh(2): Saturated soil water pressure (m) (zeroth element upper layer, first element lower layer).
    * sthu1: nfrozen soil moisture content of upper layer as a fraction of saturation
    * sthu2: Unfrozen soil moisture content of lower layer as a fraction of saturation.

    KWargs
  
    * l_dpsids_dsdz: Switch to calculate vertical gradient of soil suction with the
        assumption of linearity only for fractional saturation
        (consistent with the calculation of hydraulic conductivity)

    Returns 
                                                  
    * wflux: The flux of water between layers (kg/m2/s).
    * dwflux_dsthu1: The rate of change of the explicit flux with STHU1 (kg/m2/s).
    * dwflux_dsthu2: The rate of change of the explicit flux with STHU2 (kg/m2/s).

    '''

    theta = np.array([sthu1, sthu2]) 

    # initialise (not in the fortran)
    psi = np.zeros((2,)) 
    dpsi_dth = np.zeros((2,)) 

    for n in (0,1):
        if theta[n] <= 0.01:  # Prevent blow up for dry soil.
            psi[n] = sathh[n]/(0.01 ** b[n])
            dpsi_dth[n] = 0.0
        else:
            psi[n] = sathh[n] / (theta[n] ** b[n])
            dpsi_dth[n] = -b[n] * psi[n] / theta[n]

    # Estimate the fractional saturation at the layer boundary by
    # interpolating the soil moisture.

    thetak=(dz2*theta[0]+dz1*theta[1])/(dz2+dz1)

    '''
    In the fortran code:
!!!
!!! THIS LINE IS REPLACED WITH THE ONE FOLLOWING
!!! IN ORDER TO OBTAIN BIT-COMPARABILITY
!!!        BK=(DZ2*B(I,1)+DZ1*B(I,2))/(DZ2+DZ1)
  bk=b(i,1)
'''
    bk=b[0]

    if l_dpsids_dsdz:
        sathhm=(dz2*sathh[0]+dz1*sathh[1])/(dz2+dz1)
        #   Extra calculations for calculation of dPsi/dz as dPsi/dS * dS/dz.
        #   Split off very dry soil to avoid numerical overflows.
        if thetak < 0.01:
            dpsimdz=-bk * sathhm *                       \
                0.01**(-bk-1) *                          \
                (theta[1] - theta[0]) * (2.0/(dz2+dz1))
        else:
            dpsimdz=-bk * sathhm * thetak**(-bk-1) *     \
                (theta[1] - theta[0]) * (2.0/(dz2+dz1))

    dthk_dth1=dz2/(dz1+dz2)
    dthk_dth2=dz1/(dz2+dz1)

    # Calculate the hydraulic conductivities for transport between layers.

    k, dk_dthk = hydraulic_conductivity(bk, ks, thetak) 

    # Calculate the Darcian flux from the upper to the lower layer.

    pd=(2.0*(psi[1]-psi[0])/(dz2+dz1)+1)
    wflux=k*pd

    # Calculate the rate of change of WFLUX with respect to the STHU1 and
    # STHU2.
    dk_dth1=dk_dthk*dthk_dth1
    dk_dth2=dk_dthk*dthk_dth2
    if l_dpsids_dsdz: 
        dwflux_dsthu1=dk_dth1*(dpsimdz+1.0)-   \
            2.0*k*dpsi_dth[0]/(dz1+dz2)
        dwflux_dsthu2=dk_dth2*(dpsimdz+1.0)+   \
            2.0*k*dpsi_dth[1]/(dz1+dz2)
    else:
        dwflux_dsthu1=dk_dth1*pd-2*k*dpsi_dth[0]/(dz1+dz2)
        dwflux_dsthu2=dk_dth2*pd+2*k*dpsi_dth[1]/(dz1+dz2)

    return wflux, dwflux_dsthu1, dwflux_dsthu2


def soil_hydrology(sthu, bexp=None, ksz=None, dz=None, v_sat=None, ext=None, 
                   timestep=None, sathh=None, fw=None, 
                   l_soil_sat_down=False, l_dpsids_dsdz=False):
    '''
    Following calculation in soil_hyd_jls_mod.F90@vn4.8
    but assuming topmodel switched off
    and there's no frozen content     
    
    Note that in the fortran the elements of bexp, dz, ext, sathh, v_sat, sthu, dsmcl are 1...nshyd
    whereas here they are 0...nshyd-1.
    In the fortran, the elements of w_flux are 0...nshyd. Here, we use
    w_flux 0...nshyd-1 to be equivalent to the the fortran elements 1...nshyd and then store the fortran
    w_flux(0) in a separate python variable w_flux_surf.
    In the fortran the elements of ksz are 0...nshyd but ksz(0) doesn't get used.
    so here we put the fortran ksz(1:nshyd) here into ksz elements 0...nshyd-1.
    (qbase_l,zdepth not used here because we're assuming the top model is switched off, so don't need to 
    worry about its indices.)
        
    Args
    * sthu: Unfrozen soil moisture content of each layer as a fraction of saturation.

    KWargs

    * timestep: Model timestep (s).
    * l_soil_sat_down: Direction of super-sat soil moisture                    
    * bexp(nshyd): Clapp-Hornberger exponent.
    * dz(nshyd): Thicknesses of the soil layers (m).
    * ext(nshyd): Extraction of water from each soil layer (kg/m2/s).
    * sathh(nshyd): Saturated soil water pressure (m).
    * v_sat(nshyd): Volumetric soil moisture concentration at saturation (m3 H2O/m3 soil).
    * ksz(nshyd): Saturated hydraulic conductivity in each soil layer (kg/m2/s).
        Note this is a different length (nshyd) to the one in JULES (nshyd+1).
    * fw: Throughfall from canopy plus snowmelt minus surface runoff (kg/m2/s).   
    * l_dpsids_dsdz: Switch to calculate vertical gradient of soil suction with the
        assumption of linearity only for fractional saturation
        (consistent with the calculation of hydraulic conductivity)

    returns

    * sthu: updated sthu
    * dsmcl: Soil moisture increment (kg/m2/timestep).
        (This is not returned by the fortran subroutine but is better for code testing.
    * w_flux: fluxes between layers. w_flux[0] is the downward flux out of the top layer
        w_flux[nshyd - 1] is the downward flux out of the bottom layer 
    The fortran subroutine also returns smclsat and w_flux_surf. 

    '''
    nshyd = len(sthu)

    # initialise (not in the fortran)
    w_flux = np.zeros((nshyd,))  # The fluxes of water between layers (kg/m2/s).
    a = np.zeros((nshyd,))
    b = np.zeros((nshyd,))
    c = np.zeros((nshyd,))
    d = np.zeros((nshyd,)) 
    dsmcl = np.zeros((nshyd,)) 

    smclsat = RHO_WATER * dz * v_sat  # The saturation moisture content of each layer (kg/m2).
    smclu = sthu * smclsat

    smcl = smclu.copy() # not in JULES

    dsthumin = - sthu
    dsthumax = 1.0 - smcl /smclsat
    dwflux_dsthu1 = np.zeros((nshyd,)) 
    dwflux_dsthu2 = np.zeros((nshyd,))

    # top boundary condition
    w_flux_surf = fw

    # Calculate the Darcian fluxes and their dependencies on the soil
    # moisture contents.
    
    w_flux[nshyd-1],dwflux_dsthu1[nshyd-1] = hydraulic_conductivity(
        bexp[nshyd-1], ksz[nshyd-1], sthu[nshyd-1])

    for n in range(1, nshyd):
        w_flux[n-1], dwflux_dsthu1[n-1], dwflux_dsthu2[n-1] = darcy(
            bexp[n-1:n+1], dz[n-1], dz[n], ksz[n-1], sathh[n-1:n+1], 
            sthu[n-1], sthu[n], l_dpsids_dsdz=l_dpsids_dsdz)

    #-----------------------------------------------------------------------
    # Calculate the explicit increments.
    # This depends on the direction in which moisture in excess of
    # saturation is pushed (down if L_SOIL_SAT_DOWN, else up).
    #-----------------------------------------------------------------------
    if l_soil_sat_down:
      #-----------------------------------------------------------------------
      # Moisture in excess of saturation is pushed down.
      #-----------------------------------------------------------------------
      for n in range(nshyd):
        if n == 0:
          dsmcl[n]=(w_flux_surf-w_flux[n]-ext[n])*timestep
        else:
          dsmcl[n]=(w_flux[n-1]-w_flux[n]-ext[n])*timestep
      #-----------------------------------------------------------------------
      # Limit the explicit fluxes to prevent supersaturation.
      #-----------------------------------------------------------------------
          if dsmcl[n] >  (smclsat[n]-smcl[n]):
            dsmcl[n]=smclsat[n]-smcl[n]
            if n == 0:
                w_flux[n]=w_flux_surf-dsmcl[n]/timestep-ext[n]
            else:
                w_flux[n]=w_flux[n-1]-dsmcl[n]/timestep-ext[n]

    else:  #   NOT l_soil_sat_down
      #-----------------------------------------------------------------------
      # Moisture in excess of saturation is pushed up.
      #-----------------------------------------------------------------------
      for n in range(nshyd-1, -1, -1):  # [nshyd-1,...,0]
          if n == 0:
              dsmcl[n]=(w_flux_surf-w_flux[n]-ext[n])*timestep
          else:
              dsmcl[n]=(w_flux[n-1]-w_flux[n]-ext[n])*timestep

      #-----------------------------------------------------------------------
      # Limit the explicit fluxes to prevent supersaturation.
      #-----------------------------------------------------------------------
          if dsmcl[n] >  (smclsat[n]-smcl[n]):
            dsmcl[n]=smclsat[n]-smcl[n]
            if n == 0:
              w_flux_surf=dsmcl[n]/timestep+w_flux[n]+ext[n]
            else:
              w_flux[n-1]=dsmcl[n]/timestep+w_flux[n]+ext[n]

    #-----------------------------------------------------------------------
    # Calculate the matrix elements required for the implicit update.
    #-----------------------------------------------------------------------

    gamcon = GAMMA_W *timestep/smclsat[0]
    a[0]=0.0
    b[0]=1.0+gamcon*dwflux_dsthu1[0]
    c[0]=gamcon*dwflux_dsthu2[0]
    d[0]=dsmcl[0]/smclsat[0]

    for n in range(1,nshyd):
        gamcon = GAMMA_W *timestep/smclsat[n]
        a[n]=-gamcon*dwflux_dsthu1[n-1]
        b[n]=1.0-gamcon*(dwflux_dsthu2[n-1]-dwflux_dsthu1[n])
        c[n]=gamcon*dwflux_dsthu2[n]
        d[n]=dsmcl[n]/smclsat[n]

    #-----------------------------------------------------------------------
    # Solve the triadiagonal matrix equation.
    #-----------------------------------------------------------------------
    dsthu = gauss(a,b,c,d,dsthumin,dsthumax)

    #-----------------------------------------------------------------------
    # Diagnose the implicit fluxes.
    #-----------------------------------------------------------------------
    for n in range(nshyd):
        dsmcl[n]=dsthu[n]*smclsat[n]
        if n == 0:
          w_flux[n]=w_flux_surf-ext[n]-dsmcl[n]/timestep
        else:
          w_flux[n]=w_flux[n-1]-ext[n]-dsmcl[n]/timestep

    #-----------------------------------------------------------------------
    # Update the prognostic variables.
    #-----------------------------------------------------------------------

    smclu += dsmcl
    smcl += dsmcl
    sthu = smclu / smclsat

    return sthu, dsmcl, w_flux


def dry_down_plots():
    '''
    Example of calling the soil_hydrology function 
    '''

    dz = np.array([0.1,0.25,0.65,2.0])

    n = len(dz)
    v_sat = np.full((n,), 0.46)
    bexp = np.full((n,), 11.37)
    ksz = np.full((n,), 0.00151)
    sathh = np.full((n,), 0.32)
    sm_wilt = np.full((n,), 0.26)

    l_dpsids_dsdz = False
    l_soil_sat_down = True

    timestep = 3600

    fw = 0.0
    #ext = np.full((n,), 0.0)

    n_timesteps = 100000
    #n_timesteps = 10000
    sthu = np.zeros((n_timesteps, n))
    sthu[0, :] = 1.0
    
    for ts in range(1, n_timesteps):
        ext = np.zeros((n,))
        #ext = 0.001 * ( sthu[ts - 1, :] * v_sat[:] - sm_wilt[:]) # unrealistic
        ext[0] = 0.00001 * (sthu[ts - 1, 0] * v_sat[0]) # unrealistic
        sthu[ts, :], dsmcl, w_flux = soil_hydrology(
            sthu[ts - 1, :], bexp=bexp, ksz=ksz, dz=dz, v_sat=v_sat, ext=ext, 
            timestep=timestep, sathh=sathh, fw=fw, 
            l_soil_sat_down=l_soil_sat_down, l_dpsids_dsdz=l_dpsids_dsdz)

    vol_sm = sthu * v_sat
    
    for i in range(n):
        plt.plot(vol_sm[:, i], label=str(i + 1))
        
    plt.legend(fontsize=10, title='soil level')
    plt.savefig('dry_down.pdf')
    plt.close()
    
    return
    

def main():

    dry_down_plots()
    
    return
     

if __name__ == '__main__':

    main()
