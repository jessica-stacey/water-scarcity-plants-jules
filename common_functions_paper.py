##########################################################
# Common functions used multiple times for paper plots
##########################################################
import iris
import numpy as np

###############################
#### Formatting functions #####
###############################

def convert_co2_mmr_to_ppm_cube(cube):
    # convert to ppm
    cube_data_mmr = cube.data

    # Constants
    M_air = 28.97  # Molar mass of dry air in g/mol
    M_CO2 = 44.01  # Molar mass of CO2 in g/mol

    # Calculate ppm
    cube_data_ppm = (cube_data_mmr * M_air / M_CO2) * 1e6
    cube.data = cube_data_ppm
    return cube

def load_co2_mmr():
    '''
    Load and process the CO2 data from the processed JULES output files
    Returns: Iris cube with CO2 data in ppm
    '''
    rcp6_fname = '/data/users/jstacey/processed_jules_output/co2_rcp60.nc'
    hist_fname = '/data/users/jstacey/processed_jules_output/co2_historical.nc'
    rcp6_cube = iris.load_cube(rcp6_fname)
    hist_cube = iris.load_cube(hist_fname)
    hist_cube = hist_cube.extract(
        iris.Constraint(time=lambda cell: 1900 <= cell.point <= 2005))  # to remove 2006 to avoid duplicates with RCP60

    outcubelist = iris.cube.CubeList([hist_cube, rcp6_cube])
    cube = outcubelist.concatenate_cube()

    cube = convert_co2_mmr_to_ppm_cube(cube)

    return cube

def get_var_title(var):
    # Get nice variable title for plot labels
    var_title_dict = {'runoff': 'Total runoff',
                      'surf_roff': 'Surface runoff',
                      'sub_surf_roff': 'Sub-surface runoff',
                      'runoff_ratio': 'Runoff ratio', # not using as looks exactly same as runoff!
                      'precip': 'Precipitation',
                      'co2_mmr': 'Atmospheric CO$_2$',
                      't1p5m_gb': 'Temperature at 1.5m',
                      'tstar_gb': 'Sfc Temp',
                      'et_stom_gb': 'Transpiration',
                      'esoil_gb': 'Soil ET',
                      'fqw_gb': 'Surface moisture flux',
                      'gs': 'Stomatal conductance',
                      'rflow': 'River flow',
                      'smc_tot': 'Soil moisture',
                      'lai_gb': 'Leaf Area Index',
                      'gpp_gb': 'GPP',
                      'wsi': 'Water Scarcity Index (WSI)',
                      'supply': 'Water supply (m$^3$/day)',#(x1$0^5$)
                      'demand': 'Water demand (m$^3$/day)'}

    return var_title_dict[var]

def get_unit_title(var):
    # Get unit labels for plot titles
    var_title_dict = {'runoff': 'mm/day',
                      'surf_roff': 'mm/day',
                      'sub_surf_roff': 'mm/day', # not using as looks exactly same as runoff!
                      'precip': 'mm/day',
                      'co2_mmr': 'ppm',
                      'et_stom_gb': 'mm/day',
                      'tstar_gb': '°C',
                      't1p5m_gb': '°C',
                      #'esoil_gb': '?',
                      'fqw_gb': 'mm/day',
                      'gs': 'm/s',
                      #'rflow': '?',
                      'smc_tot': 'mm',
                      'lai_gb': 'LAI',
                      #'gpp_gb': '?'
                      }

    return var_title_dict[var]


def get_experiment_label(experiment):
    # Get experiment labels for plots
    experiment_dict = {'co2_triffid_fix': 'S1. CLIM: STOM',
            'co2_fix_noLUC': 'S2. CLIM: STOM+VEG',
            'triffid_fix': 'S3. CLIM+CO2: STOM',
            'all_noLUC': 'S4. CLIM+CO2: STOM+VEG'
            }
    return experiment_dict[experiment]

def get_contr_factor_label(factor):
    # Get labels for isolated factors for timeseries plot legends
    factor_dict = {'STOM': 'S3-S1. CO2: STOM',
                   'PLANT_PHYS': 'S4-S2. CO2: STOM+VEG',
                   'VEG_DIST': 'S2-S1. CLIM: VEG',
                   'PLANT_PHYS_VEG': 'S4-S1. CO2: STOM & CLIM+CO2: VEG'
                   }
    return factor_dict[factor]

def get_contr_factor_label_maps(factor):
    # Get labels for isolated factors for map plots - uses new lines to fit in plot
    factor_dict = {'CLIM': 'S1. CLIM:\nSTOM',
                   'co2_triffid_fix': 'S1. CLIM:\nSTOM',
                   'STOM': 'CO2: STOM',
                   'PLANT_PHYS': 'CO2:\nSTOM+VEG',
                   'VEG_DIST': 'CLIM: VEG',
                   'PLANT_PHYS_VEG': 'CO2: STOM\n&\nCLIM+CO2:\nVEG',
                   'ALL': 'S4.\nCLIM+CO2:\nSTOM+VEG',
                   'all_noLUC': 'S4.\nCLIM+CO2:\nSTOM+VEG'
                   }
    return factor_dict[factor]

def get_experiment_color_dict(experiment):
    # Get consistent color for each experiment
    color_dict = {'co2_triffid_fix': 'mediumblue',
                  'co2_fix_noLUC': 'purple',
                  'triffid_fix': 'dimgrey', #'purple',
                  'all_noLUC': 'black'
                  }
    return color_dict[experiment]
def get_contr_factor_color_dict(contr_factor):
    # Get consistent color for each contributing factor
    color_dict = {'VEG_DIST': 'forestgreen',
                  'CLIM: VEG': 'forestgreen',
                  'PLANT_PHYS': 'maroon',
                  'CO2: STOM+VEG': 'maroon',
                  'STOM': 'orangered',
                  'CO2: STOM': 'orangered',
                  'PLANT_PHYS_VEG': 'red',
                  'CO2: STOM & CLIM+CO2: VEG': 'red'
                  }
    return color_dict[contr_factor]


def get_select_basins():
    # List of basins and their PFAF IDs used in Figure 10
    return {127: 'Southern_Africa',
                          142: 'NW_Africa',
                          172: 'Nile',
                          232: 'NW_Europe',
                          294: 'Tigris-Euphrates',
                          434: 'Yangtze',
                          452: 'Himalayas',
                          453: 'India',
                          563: 'E cot Aus',
                          642: 'S Brazil',
                          742: 'Mississippi',
                          }

#############################################
## Common functions for masking Iris cubes #####
#############################################

def mask_cube_by_ar6_region(cube, region):
    '''
    Mask iris cube by IPCC AR6 regions
    Args:
        cube: Iris cube to mask
        region: IPCC AR6 region name

    Returns: Iris cube now masked by the selected region

    '''
    import regionmask
    import numpy as np

    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    ar6_regions = regionmask.defined_regions.ar6.land
    print('Masking cube for AR6 region {}'.format(region.name))

    # for region in giorgi:
    mask = ar6_regions.mask(lon_grid, lat_grid)
    single_region_mask = mask == region.number

    data_shape = cube.shape
    broadcasted_mask = np.broadcast_to(single_region_mask, data_shape)

    # Apply the mask to the cube data
    masked_data = np.ma.masked_where(~broadcasted_mask, cube.data)
    masked_cube = cube.copy()
    masked_cube.data = masked_data

    return masked_cube

def apply_basin_mask_to_cube(cube, PFAF_ID):
    '''
    Mask cube by basin using PFAF_ID
    Args:
        cube: Iris cube to mask
        PFAF_ID: Hydrobasin unique PFAF ID as integer

    Returns: Iris cube now masked by the selected basin

    '''
    import regionmask
    import geopandas as gpd
    import numpy as np
    import numpy.ma as ma

    # read in shape file and select single basin
    print('Masking cube for PFAF_ID {}'.format(PFAF_ID))
    shpfile = '/data/users/jstacey/hydrobasins/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp'
    basins = gpd.read_file(shpfile)
    basin = basins[basins['PFAF_ID'] == PFAF_ID]

    # get mask
    basins_regions = regionmask.Regions(basin.geometry)

    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points

    # Rasterises the regions to the lon, lat grid
    mask = basins_regions.mask(lon, lat)
    mask_cube = mask.to_iris()

    # Broadcast the mask to the shape of the input cube
    mask_array = np.broadcast_to(mask_cube.data, cube.shape)
    mask_array = ma.masked_invalid(mask_array)

    # apply numpy mask to new cube
    masked_cube_data = np.ma.masked_where(mask_array, cube.data)
    cube_basin = cube.copy()
    cube_basin.data = masked_cube_data

    return cube_basin

#### Functions for calculating contributing factors

def calc_factor_inf_df(df, percent_diff=False):
    '''
    Calculate the contribution of each factor to the total change in WSI using Pandas dataframes,
    by taking difference between simulations
    Args:
        df: Pandas dataframe with columns for each factor
        percent_diff: For output in percentage change

    Returns: Pandas dataframe with additional columns for each factor

    '''
    diff_dict = {'CO2: STOM': ['co2_triffid_fix', 'triffid_fix'],
             'CLIM: VEG': ['co2_triffid_fix', 'co2_fix_noLUC'],
             'CO2: STOM+VEG': ['co2_fix_noLUC', 'all_noLUC'],
             'CO2: STOM & CLIM+CO2: VEG': ['co2_triffid_fix', 'all_noLUC']}

    for factor in diff_dict:
        col_name_0 = '{}'.format(diff_dict[factor][0])
        col_name_1 = '{}'.format(diff_dict[factor][1])

        diff = df[col_name_1] - df[col_name_0]
        if percent_diff:
            diff = (diff / df[col_name_0]) * 100
        df[factor] = diff
    return df

def calc_contr_factors_timeseries(cube_dict, var_list, calc_rel_diff=False):
    '''
    Calculate the contribution of each factor to the total change in WSI using Iris cubes,
    by taking difference between simulations
    Args:
        cube_dict: dictionary of Iris cubes for each variable and simulation e.g., dict_name[var_name, simulation_name]
        var_list: list of variables to calculate contributing factors for
        calc_rel_diff: To output the % differnce between the simulations

    Returns: Dictionary of Iris cubes for each variable and contributing factor

    '''
    from iris.analysis import maths as ia_maths

    abs_diff_dict = {}
    print('Calculating contribution of each factor')

    for var in var_list:
        abs_diff_dict[var, 'STOM'] = ia_maths.subtract(cube_dict[var, 'triffid_fix'], cube_dict[var, 'co2_triffid_fix'])
        abs_diff_dict[var, 'PLANT_PHYS'] = ia_maths.subtract(cube_dict[var, 'all_noLUC'], cube_dict[var, 'co2_fix_noLUC'])
        abs_diff_dict[var, 'VEG_DIST'] = ia_maths.subtract(cube_dict[var, 'co2_fix_noLUC'],cube_dict[var, 'co2_triffid_fix'])
        abs_diff_dict[var, 'PLANT_PHYS_VEG'] = ia_maths.subtract(cube_dict[var, 'all_noLUC'], cube_dict[var, 'co2_triffid_fix'])

    if calc_rel_diff:
        print('Calculating contributing factor for each variable and factor')
        rel_contr_dict = {}

        for (var, contr_factor) in abs_diff_dict:
            contr_factor_cube = abs_diff_dict[var, contr_factor]
            if contr_factor == 'STOM':
                rel_contr_factor_cube = ia_maths.divide(contr_factor_cube, cube_dict[var, 'co2_triffid_fix'])
            elif contr_factor == 'PLANT_PHYS':
                rel_contr_factor_cube = ia_maths.divide(contr_factor_cube, cube_dict[var, 'co2_fix_noLUC'])
            elif contr_factor == 'VEG_DIST':
                rel_contr_factor_cube = ia_maths.divide(contr_factor_cube, cube_dict[var, 'co2_triffid_fix'])
            elif contr_factor == 'PLANT_PHYS_VEG':
                rel_contr_factor_cube = ia_maths.divide(contr_factor_cube, cube_dict[var, 'co2_triffid_fix'])
            else:
                print('Contr_factor {} not included'.format(contr_factor))

            rel_contr_dict[var, contr_factor] = ia_maths.multiply(rel_contr_factor_cube, 100) # to get %

        return rel_contr_dict
    else:
        return abs_diff_dict

def calc_contr_factors_map(cube_dict, var_list, rel_diff=False):
    '''
    Calculate the contribution of each factor to the total change in WSI using Iris cubes,
    by taking difference between simulations
    Args:
        cube_dict: dictionary of Iris cubes for each variable and simulation e.g., dict_name[var_name, simulation_name]
        var_list: list of variables to calculate contributing factors for
        rel_diff: To output the % differnce between the simulations

    Returns: Dictionary of Iris cubes for each variable and contributing factor

    '''
    from iris import analysis as ia
    from iris.analysis import maths as ia_maths

    diff_dict = {}
    print('Calculating contribution of each factor')
    for var in var_list:
        diff_dict[var, 'CLIM'] = cube_dict[var, 'co2_triffid_fix']
        diff_dict[var, 'PLANT_PHYS'] = cube_dict[var, 'all_noLUC'] - cube_dict[var, 'co2_fix_noLUC']
        diff_dict[var, 'VEG_DIST'] = cube_dict[var, 'co2_fix_noLUC'] - cube_dict[var, 'co2_triffid_fix']
        diff_dict[var, 'PLANT_PHYS_VEG'] = cube_dict[var, 'all_noLUC'] - cube_dict[var, 'co2_triffid_fix']
        diff_dict[var, 'STOM'] = cube_dict[var, 'triffid_fix'] - cube_dict[var, 'co2_triffid_fix']
        diff_dict[var, 'ALL'] = cube_dict[var, 'all_noLUC']

    if rel_diff:
        print('Calculating the relative diference')
        #contr_factor_cube = diff_dict[var, contr_factor]
        diff_dict[var, 'STOM'] = (diff_dict[var, 'STOM'] / cube_dict[var, 'co2_triffid_fix'])* 100
        diff_dict[var, 'PLANT_PHYS'] = (diff_dict[var, 'PLANT_PHYS'] / cube_dict[var, 'co2_fix_noLUC']) * 100
        diff_dict[var, 'VEG_DIST'] = (diff_dict[var, 'VEG_DIST'] / cube_dict[var, 'co2_triffid_fix']) *100
        diff_dict['PLANT_PHYS_VEG'] = (diff_dict[var, 'PLANT_PHYS_VEG'] / cube_dict[var, 'co2_triffid_fix'])* 100

    return diff_dict

#########################
### Plot formatting #####
#########################

def position_cbar_colno_dependant(n, col_no):
    '''
    Position the colorbar based on the number of columns in the plot - used for Fig. 4
    Args:
        n: total number of columns
        col_no: column number

    Returns: Loction for cbar

    '''
    left_col = 0.1
    gap = 0.01 # was 0.2
    width = (1 - left_col - 8*gap)/n
    cbar_loc = [left_col + (col_no*width) + ((col_no*2+1) * gap), 0.05, width, 0.02]
    return cbar_loc

def add_cbar(fig, mappable, cbar_title, var, cbar_axes_loc=None):
    '''
    Add colorbar to plot - used in Fig. 4
    Args:
        fig: matplotlib figure
        mappable: your plot
        cbar_title: title for colorbar
        var: variable string
        cbar_axes_loc: list, axes location for colorbar

    Returns: colorbar

    '''
    import matplotlib.pyplot as plt

    if cbar_axes_loc is not None:
        cbar_axes = fig.add_axes(cbar_axes_loc)
    else:
        cbar_axes = fig.add_axes([0.2, 0.15, 0.6, 0.05]) # left, bottom, width, height
    cbar = plt.colorbar(mappable, cbar_axes, orientation='horizontal')
    cbar.set_label('{}'.format(cbar_title), fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    if var == 'gs':
        cbar.set_ticks([-0.002, -0.001, 0, 0.001, 0.002])
        cbar.set_ticklabels([-0.002, -0.001, 0, 0.001, 0.002])
    return cbar

#########################
##### For basin plots ###
#########################

def get_basin_colour_dict(col, reldiff=False, seasonal_flag=False):
    '''
    Get color dictionary for basin plots
    Args:
        col: Name of simulation or isolated factor
        reldiff: % difference flag
        seasonal_flag: for seasonal Fig. S6

    Returns: Dictionary of color ranges with value range and colour

    '''
    if col in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC', 'CLIM', 'ALL']:
        print('Finding   color for {}'.format(col))
        color_dict = {(5, 10000000000): 'dimgrey',
                      (1, 5): 'darkred',  # dimgrey
                      (0.4, 1): 'saddlebrown',
                      (0.3, 0.4): 'red',
                      (0.2, 0.3): 'orangered',
                      (0.1, 0.2): 'salmon',
                      (0.05, 0.1): 'peachpuff',
                      (0, 0.05): 'white',
                      }

    elif reldiff and seasonal_flag==False:  # for reldiff contr factor changes - change 100s back to 60 for ann
        color_dict = {(40, 60): 'darkred',
                      (30, 40): 'saddlebrown',
                      (20, 30): 'red',
                      (10, 20): 'lightsalmon',
                      (5, 10): 'peachpuff',
                      (-5, 5): 'white',
                      (-10, -5): 'paleturquoise',
                      (-20, -10): 'darkturquoise',
                      (-30, -20): 'teal',
                      (-40, -30): 'darkslategray',
                      (-70, -40): 'midnightblue',
                      }
    elif reldiff and seasonal_flag: # For Fig. S6
        color_dict = {(40, 100): 'darkred',
                      (30, 40): 'saddlebrown',
                      (20, 30): 'red',
                      (10, 20): 'lightsalmon',
                      (5, 10): 'peachpuff',
                      (-5, 5): 'white',
                      (-10, -5): 'paleturquoise',
                      (-20, -10): 'darkturquoise',
                      (-30, -20): 'teal',
                      (-40, -30): 'darkslategray',
                      (-90, -40): 'midnightblue',
                      }

    return color_dict


def make_color_list_for_wsi(df, col, color_dict):
    '''
    Make a list of colors representing each value in the dataframe columnn used for basin plots
    Args:
        df: Pandas dataframe chich includes a column with name col
        col: Column name to get colors for
        color_dict: dictionary of value ranges and colours assigned in get_basin_colour_dict() above

    Returns: Luist of colours  to assign to new column in df

    '''
    from matplotlib import pyplot as plt

    color_li = []

    print('{} has max {}'.format(col, df[col].max()))
    print('{} has min {}'.format(col, df[col].min()))

    for val in df[col]:
        for (range_min, range_max) in color_dict:
            color = color_dict[(range_min, range_max)]
            if range_min == range_max:  # when looking for value that matches number exactly e.g. 0
                if val == range_min:
                    color_li.append(color)
            elif val >= range_min and val < range_max:
                color_li.append(color)

    if len(color_li) != len(df[col]):
        fig, ax = plt.subplots(figsize=(20, 15))
        plt.hist(color_li)
        plt.savefig('/home/h06/jstacey/MSc/logs/color_hist.png', dpi=100, bbox_inches='tight', facecolor='white')
        raise ValueError('length of new color list ({}) and dataframe column are unequal ({}). If color list is longer '
                         'this is likely an error with the color dict; if its shorter, its likely because the ranges'
                         'miss numbers (i.e., extend min and max) '.format(len(color_li), len(df[exp])))
    return color_li

def get_basin_colour_dict_month_count(col, reldiff=False):
    '''
    Get color dictionary for basin plots for month count Fog. S5
    Args:
        col: simulation or isolated factor name
        reldiff: True for % difference

    Returns: Dictionary of color ranges with value range and colour

    '''
    if col in ['co2_triffid_fix', 'all_noLUC']:
        color_dict = {(10, 13): 'dimgrey',
                     (8, 10): 'darkred',
                     (6, 8): 'saddlebrown',
                     (4, 6): 'red',
                     (3, 4): 'orangered',
                     (2, 3): 'salmon',
                     (1, 2): 'peachpuff',
                     (0, 1): 'white',
                     }
    else:
        color_dict = {(2, 3): 'saddlebrown',
                         (1, 2): 'red',
                         (0.5, 1): 'lightsalmon',
                         (-0.5, 0.5): 'white',
                         (-1, -0.5): 'darkturquoise',
                         (-2, -1): 'teal',
                         (-3, -2): 'darkslategray',
                         (-4, -3): 'midnightblue',
                     }


    return color_dict


def get_fut_pop_count_by_basin(PFAF_ID, in_millions=False):
    '''
    Get future population count by basin
    Args:
        PFAF_ID: Unique hydrobasin ID
        in_millions: True for value in millions

    Returns: Float, population count for that basin

    '''
    import pandas as pd

    # To load pop_df from csv
    df_pop = pd.read_csv('/home/h06/jstacey/MSc/csv_files/table2_df_pop_by_basin-2076-2095.csv')
    df_pop = df_pop.drop('Unnamed: 0', axis=1)
    pop_count = df_pop.loc[df_pop['PFAF_ID'] == PFAF_ID, 'pop_count']
    pop_count = round(pop_count.item()) # round to nearest integer
    #print('Pop count is {:,}'.format(pop_count))
    if in_millions:
        pop_count = pop_count/1000000
        pop_count = round(pop_count)
        #print('Pop count is {:,}M'.format(pop_count))
    return pop_count

def sum_pop_in_multiple_basins(basin_list):
    '''
    Sum population in multiple basins
    Args:
        basin_list: list of PFAF_IDs

    Returns: float, total population count

    '''
    total_pop = 0
    for id in basin_list:
        pop_count = get_fut_pop_count_by_basin(id, in_millions=False)
        total_pop += pop_count
    return total_pop

def calc_basins_incr_decr(df, factor, threshold, table_dict):
    '''
    Calculate the number of basins and population count in basins which have increased or decreased by a certain threshold
    used for Table 2
    Args:
        df: Pandas Dataframe which has column name same asa 'factor'
        factor: str, column name - simualtion or isolated factor
        threshold: float, threshold in which to calculate increase or decrease
        table_dict: dictionary of table 2 - defined in table2_output_basins_pop_incr_decr_to_csv()

    Returns: table_dict with updated values

    '''
    df_threshold_pos = df[(df[factor] > threshold)]
    df_threshold_neg = df[(df[factor] < -threshold)]

    incr_pfaf_list = df_threshold_pos['PFAF_ID'].tolist()
    decr_pfaf_list = df_threshold_neg['PFAF_ID'].tolist()

    incr_pop = sum_pop_in_multiple_basins(incr_pfaf_list)
    decr_pop = sum_pop_in_multiple_basins(decr_pfaf_list)

    table_dict['Factor'].append(factor)
    table_dict['Threshold'].append(threshold)
    table_dict['basin_ct_incr'].append(len(incr_pfaf_list))
    table_dict['basin_ct_decr'].append(len(decr_pfaf_list))
    table_dict['pop_ct_incr'].append(incr_pop)
    table_dict['pop_ct_decr'].append(decr_pop)
    tot_fut_pop = 9305893888 # calculated in common.calc_tot_pop()
    table_dict['pop_perc_incr'].append(round((incr_pop / tot_fut_pop) * 100, 1))
    table_dict['pop_perc_decr'].append(round((decr_pop / tot_fut_pop) * 100, 1))

    return table_dict