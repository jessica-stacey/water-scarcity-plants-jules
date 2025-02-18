# -*- coding: iso-8859-1 -*-

'''
Code to compare the shape of the Sinclair water stress parametrisation and the 
JULES parametrisation with various soil textural classes.
'''

import os

import numpy as np
import matplotlib.pyplot as plt

import soil_cosby_parameters

from soil_water_stress_factors import *


def compare_sinclair_and_jules_p0(plotfolder=None):
    '''
    Plots the Sinclair and JULES functions for each soil textural
    class as a separate plot on one figure. 
    '''

    # Create array of soil water values, as percentages.
    soil_water_percent = np.arange(0.01, 70.0, 0.01)

    fig = plt.figure(figsize=[8.0*1.5, 6.0*1.5])
    
    ax = fig.add_subplot(1, 1, 1) # one big plot to cover whole thing
    
    # Turn off axis lines and ticks of the big subplot
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    # Set common labels
    ax.set_xlabel('fraction of transpirable soil water')
    ax.set_ylabel(r'soil water stress factor $\beta$')

    nx = 4 # number of columns of plots
    ny = 3 # number of rows of plots
    
    # loop through each soil textural class
    for i, soil_textural_class in enumerate(soil_cosby_parameters.SOIL_TEXTURAL_CLASS):
        ax = fig.add_subplot(ny, nx, i + 1)

        soil = soil_cosby_parameters.SoilType(soil_textural_class=soil_textural_class)

        # Fraction of transpirable soil water
        ftsw = (soil_water_percent / 100.0 - soil.jules_soil_parameters['sm_wilt']) \
            / (soil.jules_soil_parameters['sm_crit'] - soil.jules_soil_parameters['sm_wilt'])

        # Calculate and plot the Sinclair factors
        # These values of psi_e are the same as Verhoef et al 2014 Fig 2c.
        beta_sinclair_low = sinclair_soil_factor(soil_water_percent, soil, psi_e_in_Pa=-1.0E6)
        beta_sinclair_middle = sinclair_soil_factor(soil_water_percent, soil, psi_e_in_Pa=-1.5E6)
        beta_sinclair_high = sinclair_soil_factor(soil_water_percent, soil, psi_e_in_Pa=-2.0E6)
        
        plt.plot(ftsw, beta_sinclair_middle, '-', color='green', label='Sinclair,$\phi_e=$-1.5$\pm$0.5MPa', alpha=0.5)  
        plt.fill_between(ftsw, beta_sinclair_low, beta_sinclair_high, color='green', alpha=0.2)

        # Calculate and plot the JULES factors
        
        # The default JULES line is just x=y
        plt.plot(ftsw, ftsw, 'k:', label='JULES, p0=0 (default)')

        p0 = 0.5
        beta_jules = jules_soil_factor(soil_water_percent, soil, p0=p0)
        plt.plot(ftsw, beta_jules, 'b-', label='JULES, p0=' + str(p0))

        p0 = 0.6
        beta_jules = jules_soil_factor(soil_water_percent, soil, p0=p0)
        plt.plot(ftsw, beta_jules, 'b--', label='JULES, p0=' + str(p0))

        # Tidy the plot up.
        plt.title(soil_textural_class, fontsize=12)

        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plotfilename = os.path.join(plotfolder, 'compare_sinclair_and_jules_p0.' + ext)
        plt.savefig(plotfilename)

    plt.clf()

    return


def main():
    plotfolder = '.'
    compare_sinclair_and_jules_p0(plotfolder=plotfolder)
    
    
if __name__ == '__main__':
    main()
