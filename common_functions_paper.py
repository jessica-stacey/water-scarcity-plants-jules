##########################################################
# Common functions used multiple times for paper plots
##########################################################
import iris
import numpy as np

#### Formatting functions ####

def int_to_roman(num):
    # Function to convert integer to Roman numerals
    val = [
        10, 9, 5, 4,
        1
        ]
    syb = [
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while  num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def load_co2_mmr():
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

def get_var_title(var):
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
    experiment_dict = {'co2_triffid_fix': 'S1. CLIM: STOM',
            'co2_fix_noLUC': 'S2. CLIM: STOM+VEG',
            'triffid_fix': 'S3. CLIM+CO2: STOM',
            'all_noLUC': 'S4. CLIM+CO2: STOM+VEG'
            }
    return experiment_dict[experiment]

def get_contr_factor_label(factor):
    factor_dict = {'CLIM': 'CLIM',
                   'STOM': 'CO2: STOM',#  (E3-E1)/E1',
                   'PLANT_PHYS': 'CO2: STOM+VEG',#  (E4-E2)/E2',
                   'VEG_DIST': 'CLIM: VEG',#  (E2-E1)/E1',
                   'PLANT_PHYS_VEG': 'CO2: STOM & CLIM+CO2: VEG'#'CLIM+CO2: STOM+VEG',#  (E4-E1)/E1'
                   }
    return factor_dict[factor]

def get_contr_factor_label_maps(factor):
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

## Plot color dictionaries
def get_experiment_color_dict(experiment):
    color_dict = {'co2_triffid_fix': 'mediumblue',
                  'co2_fix_noLUC': 'purple',
                  'triffid_fix': 'dimgrey', #'purple',
                  'all_noLUC': 'black'
                  }
    return color_dict[experiment]
def get_contr_factor_color_dict(contr_factor):
    color_dict = {'VEG_DIST': 'forestgreen',
                  'PLANT_PHYS': 'maroon',
                  'STOM': 'orangered',
                  'PLANT_PHYS_VEG': 'red'
                  }
    return color_dict[contr_factor]


def get_contr_factor_color_dict_basins(contr_factor):
    color_dict = {'CLIM: VEG': 'seagreen',
                  'CO2: STOM+VEG': 'maroon',
                  'CO2: STOM': 'orangered',
                  'CO2: STOM & CLIM+CO2: VEG': 'red'
                  }
    return color_dict[contr_factor]

def get_select_basins():
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

## Common functions

def unify_cubes(cube_in, cube_format_to_copy):
    # cubes must have data in same shape - set something up to chack for this
    cube_in_data = cube_in.data
    cube_out = cube_format_to_copy.copy()
    cube_out.data = cube_in_data
    return cube_out


def cap_wsi_cube_abv_20(cube, threshold):
    import numpy as np
    print('Capping all WSI values > 20 to 20 for each experiment and month')

    num_values_above_threshold = np.sum(cube.data > threshold)
    total_values = cube.data.size
    percentage_changed = (num_values_above_threshold / total_values) * 100
    print(f"Percentage of values that were changed: {percentage_changed:.2f}%")

    num_values_above_threshold = np.sum(cube.data < 0)
    percentage_changed_0 = (num_values_above_threshold / total_values) * 100
    print(f"Number of negative values: {num_values_above_threshold:.2f} {percentage_changed_0:.2f}%")

    data = cube.data
    cube.data = np.clip(data, 0, threshold)  # puts cap on data

    num_values_above_threshold = np.sum(cube.data >= threshold)
    percentage_changed_after = (num_values_above_threshold / total_values) * 100

    if percentage_changed_after != percentage_changed:
        raise ValueError("The percentages before ({}) and after ({}) capping the cube data are not the same.".format(percentage_changed, percentage_changed_after))

    return cube

def mask_cube_by_ar6_region(cube, region, quickplot=False):
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
    # if region.number == 0 and quickplot:
    #    qplt.pcolormesh(masked_cube[0])
    return masked_cube

def apply_basin_mask_to_cube(cube, PFAF_ID):
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

def calc_factor_inf_df_with_wsi_colname(df, percent_diff=True):
    diff_dict = {'CO2: STOM': ['co2_triffid_fix', 'triffid_fix'],
             'CLIM: VEG': ['co2_triffid_fix', 'co2_fix_noLUC'],
             'CO2: STOM+VEG': ['co2_fix_noLUC', 'all_noLUC'],
             'CO2: STOM & CLIM+CO2: VEG': ['co2_triffid_fix', 'all_noLUC']}

    for factor in diff_dict:
        col_name_0 = 'WSI_mean_{}'.format(diff_dict[factor][0])
        col_name_1 = 'WSI_mean_{}'.format(diff_dict[factor][1])

        diff = df[col_name_1] - df[col_name_0]
        if percent_diff:
            diff = (diff/df[col_name_0])*100
        df[factor] = diff
    return df

def calc_factor_inf_df(df, percent_diff=False):
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

# Preprocessing fucntions
def set_up_comparable_cubes(cube_in, use_time_format_from_cube):
    import iris
    for vrb in ['time', 'latitude', 'longitude']:
        if cube_in.coord(vrb).has_bounds() == False:
            cube_in.coord(vrb).guess_bounds()

    time_coord = use_time_format_from_cube.coord('time')
    # time_coord.units = cf_units.Unit(time_coord.units.origin, calendar="gregorian")

    coord_list = [(time_coord, 0),
                  (cube_in.coord('latitude'), 1),
                  (cube_in.coord('longitude'), 2)]

    # Set up new cube inserting each coord
    cube_out = iris.cube.Cube(cube_in.data,
                              standard_name=cube_in.standard_name,
                              long_name=cube_in.long_name,
                              units=cube_in.units,
                              dim_coords_and_dims=coord_list)
    return cube_out

def mask_non_water_scarce_regions(wsi_cube, cube_to_mask, mask_min_value, mask_max_value):
    import numpy as np
    import numpy.ma as ma

    wsi_cube_data = wsi_cube.data

    mask1 = wsi_cube_data > mask_min_value
    mask2 = wsi_cube_data < mask_max_value
    mask3 = np.logical_and(mask1, mask2)

    mask = ma.masked_where(mask3, wsi_cube_data)
    cube_to_mask.data = np.ma.masked_where(np.ma.getmask(mask), cube_to_mask.data)
    return cube_to_mask

def agg_funcs(var):
    import iris
    agg_dict = {'supply': iris.analysis.SUM,
                'demand': iris.analysis.SUM,
                'WSI': iris.analysis.MEDIAN
                }
    return agg_dict[var]

def convert_cube_units(cube, cube_units_out):
    import iris
    if (cube.units == 'mm/day' and cube_units_out == 'm**3/day'):
        print('Converting cube units: mm/day -> m3/day')
        wghts = iris.analysis.cartography.area_weights(cube, normalize=False)  # in m**2
        wghts_cube = cube.copy(data=wghts)
        wghts_cube.units = "m**2"
        cube = cube * wghts_cube  # units are now a fraction of m**3.s**-1
        cube.convert_units("m**3/day")
    return cube

def calc_WSI_timeseries(supply_dict, demand_cube, experiment_list, spatial_agg=None, rolling_mean_years=None):
    import iris
    temp_dict = {}
    wsi_dict = {}
    for plot in ['demand', 'supply', 'WSI']:
        for experiment in experiment_list:
            print('processing cube for {}'.format(plot))
            if plot == 'supply':
                cube = supply_dict[experiment]
            elif plot == 'demand':
                cube = demand_cube

            if plot in ['supply', 'demand']:
                temp_dict[plot, experiment] = cube
            elif plot == 'WSI': # calculate annual mean WSI
                wsi_cube = temp_dict['demand', experiment] / temp_dict['supply', experiment]
                wsi_cube = wsi_cube.aggregated_by('year', iris.analysis.MEAN) # added as now calculating WSI each month
                if spatial_agg == 'median':
                    wsi_cube = wsi_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEDIAN)
                elif spatial_agg == 'mean':
                    wsi_cube = wsi_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN,
                                                  weights=iris.analysis.cartography.area_weights(wsi_cube))
                else:
                    'No spatial aggregation applied'
                if rolling_mean_years is not None:
                    wsi_cube = wsi_cube.rolling_window('year', iris.analysis.MEAN, rolling_mean_years)

                wsi_dict['wsi', experiment] = wsi_cube
            else:
                print('No spatial averaging selected')

    return wsi_dict

def calc_wsi_mean_by_ar6_region(df, exp_list, inc_factors=False):
    col_list = []

    for exp in exp_list:
        col_list.append('WSI_mean_{}'.format(exp))
    if inc_factors:
        for factor in ['CO2: STOM', 'CLIM: VEG', 'CO2: STOM+VEG', 'CLIM: VEG + CO2: STOM+VEG']:
            col_list.append('{}'.format(factor))

    mm_region = df.groupby(['region_name'], sort=True)[col_list].mean()
    return mm_region
'''
def calc_wsi_mean_by_ar6_region_from_supply_demand(df, exp_list, inc_factors=False):
    col_list = []

    for exp in exp_list:
        col_list.append('supply_mean_{}'.format(exp))

    col_list.append('demand_mean_all_noLUC')

    ann_mean = df.groupby(['year'], sort=True)[col_list].mean()

    # Calc WSI
    # Not sure whats happened here - I think got deleted, see one above?!

    return mm_region
'''

### Plot formatting
def position_cbar_colno_dependant(n, col_no):
    left_col = 0.1
    gap = 0.01 # was 0.2
    width = (1 - left_col - 8*gap)/n
    cbar_loc = [left_col + (col_no*width) + ((col_no*2+1) * gap), 0.05, width, 0.02]
    return cbar_loc

def add_cbar(fig, mappable, cbar_title, var, font_size=None, cbar_axes_loc=None):
    import matplotlib.pyplot as plt

    if cbar_axes_loc is not None:
        cbar_axes = fig.add_axes(cbar_axes_loc)
    else:
        cbar_axes = fig.add_axes([0.2, 0.15, 0.6, 0.05]) # left, bottom, width, height
    cbar = plt.colorbar(mappable, cbar_axes, orientation='horizontal')
    if font_size is None:
        font_size = 18
    cbar.set_label('{}'.format(cbar_title), fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    if var == 'gs':
        cbar.set_ticks([-0.002, -0.001, 0, 0.001, 0.002])
        cbar.set_ticklabels([-0.002, -0.001, 0, 0.001, 0.002])
    return cbar

def contourf_shift(levels, data=None, extend='neither'):
    """
    Shift the contour levels or the data to be plotted by a small amount.
    This small amount is determined by the mean difference between levels.

    We do not shift the first level, which ensures the full data range is
    still covered by the levels, e.g. making sure data values of 0 are
    covered when original minimum level is 0.

    If data is provided then we return shifted data instead of shifted
    levels.

    :arg list levels:     List of the contour levels.
    :arg np.ndarray data: Optionally choose to shift the data to be plotted.
    :arg string extend:   Used when shifting data to decide whether to fix
                          the max.

    :returns:             Shifted levels list or data array.
    :rtype:               levels or np.ndarray.

    """
    import numpy as np

    if isinstance(levels, str):
        raise ValueError('levels should be a list.')
    else:
        try:
            iter(levels)
        except TypeError:
            print('levels should be a list.')
    if data is not None and type(data) not in [
            np.ndarray, np.ma.core.MaskedArray]:
        print('data should be a numpy array.')

    diff_list = [
        levels[lev] - levels[lev - 1] for lev in range(1, len(levels))]
    shift_value = 0.0001 * np.mean(diff_list)
    if data is not None:
        data += shift_value
        # If we are looking at probabilities (where the color bar does not
        # extend past 0 or 100) we do not want any values greater than 100%.
        if extend in ['neither', 'min']:
            max_level = levels[-1]
            data[np.where(data > max_level)] = max_level
        return data
    else:
        shifted_levels = [levels[0], ]
        shifted_levels.extend([level + shift_value for level in levels[1:]])
        return shifted_levels

def _contourf_with_retry(cube, levels=None, cmap=None, colors=None, extend='neither',
                         axes=None):
    """
    There is a small chance that iplt.contourf will fail with
    'shapely.geos.TopologicalError', 'ValueError' or 'AttributeError'.
    Get the mean difference between the contour levels so we can pick a
    relatively 'small' value to shift the data by, then try again!
    (For details of this fix see:
    https://code.metoffice.gov.uk/trac/glosea/wiki/GS_PP/ContourPlotFix)

    At cartopy 0.16 a fix was introduced to handle this problem, but appears
    to have introduced a new bug which is less easy to capture. We shift the
    levels to avoid hitting this bug.

    :arg iris.cube.Cube cube: Merged cube containing data for iplt.contourf.
    :arg list levels:         List of the contour levels.
    :arg list colors:         List of contour line colours.
    :arg string extend:       Used when shifting data to decide whether to fix
                              the max.
    :arg plt.subplot axes:    Axis to use for this map.

    :returns:                 Plot with contour lines on figure
    :rtype:                   iris.plot

    """
    import shapely
    import iris.plot as iplt

    levels = contourf_shift(levels)
    try:
        im = iplt.contourf(
            cube, levels=levels, cmap=cmap, extend=extend, axes=axes)
    except (shapely.geos.TopologicalError, ValueError, AttributeError) as err:
        cube.data = contourf_shift(levels, data=cube.data, extend=extend)
        plt.cla()
        im = iplt.contourf(
            cube, levels=levels, cmap=cmap, extend=extend)
    return im

# Producing dataframes for later
def calc_median_wsi_by_ar6_region():
    import iris
    from iris import coord_categorisation
    import regionmask
    import common_functions_paper as common
    import pandas as pd

    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    monthly_demand_cube = monthly_demand_cube.extract(iris.Constraint(year=lambda cell: 2076 <= cell.point <= 2095))

    ar6_land_regions = regionmask.defined_regions.ar6.land

    # # To test with one region
    # region_number = ar6_land_regions.abbrevs.index('EAS')
    # filtered_region = ar6_land_regions[region_number]

    wsi_median_dict = {}

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Calcualting wsi for {}'.format(experiment))
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(iris.Constraint(year=lambda cell: 2076 <= cell.point <= 2095))
        monthly_wsi_cube = monthly_demand_cube / monthly_supply_cube

        for region in ar6_land_regions:
            cube_region = common.mask_cube_by_ar6_region(monthly_wsi_cube, region=region)
            cube_region_median = cube_region.collapsed(['latitude', 'longitude', 'time'], iris.analysis.MEDIAN)
            wsi_median_dict[experiment, region.name] = cube_region_median.data

    df = pd.DataFrame(list(wsi_median_dict.items()), columns=['keys', 'WSI Median'])

    # Split the 'keys' column into separate columns
    df[['Experiment', 'Region']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    df.drop(columns=['keys'], inplace=True)
    df = df.pivot(index='Region', columns='Experiment', values='WSI Median')

    df = calc_factor_inf_df_monthly(df)

    # Save the DataFrame to a CSV
    df.to_csv('/home/h06/jstacey/MSc/csv_files/wsi_monthly_gp_median_with_inf_factors_fut.csv', index=False)
    print('Dataframe saved to /home/h06/jstacey/MSc/csv_files/wsi_median.csv')
    return df

# For basin plots

def get_basin_colour_dict(col, reldiff=False):
    if col in ['co2_triffid_fix', 'all_noLUC', 'CLIM', 'ALL']:
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
    elif reldiff == False:
        color_dict = {(1, 2): 'darkred',
                      (0.5, 1): 'saddlebrown',
                      (0.2, 0.5): 'red',
                      (0.1, 0.2): 'lightsalmon',
                      (0.05, 0.1): 'peachpuff',
                      (-0.05, 0.05): 'white',
                      (-0.1, -0.05): 'paleturquoise',
                      (-0.2, -0.1): 'darkturquoise',
                      (-0.5, -0.2): 'teal',
                      (-1, -0.5): 'darkslategray',
                      (-12, -1): 'midnightblue',
                      }

    else:  # for reldiff contr factor changes - change 100s back to 60 for ann
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

    return color_dict


def make_color_list_for_wsi(df, exp, color_dict):
    from matplotlib import pyplot as plt

    color_li = []

    print('{} has max {}'.format(exp, df[exp].max()))
    print('{} has min {}'.format(exp, df[exp].min()))

    for val in df[exp]:
        for (range_min, range_max) in color_dict:
            color = color_dict[(range_min, range_max)]
            if range_min == range_max:  # when looking for value that matches number exactly e.g. 0
                if val == range_min:
                    color_li.append(color)
            elif val >= range_min and val < range_max:
                color_li.append(color)

    if len(color_li) != len(df[exp]):
        fig, ax = plt.subplots(figsize=(20, 15))
        plt.hist(color_li)
        plt.savefig('/home/h06/jstacey/MSc/logs/color_hist.png', dpi=100, bbox_inches='tight', facecolor='white')
        raise ValueError('length of new color list ({}) and dataframe column are unequal ({}). If color list is longer '
                         'this is likely an error with the color dict; if its shorter, its likely because the ranges'
                         'miss numbers (i.e., extend min and max) '.format(len(color_li), len(df[exp])))
    return color_li

def get_basin_colour_dict_month_count(col, reldiff=False):
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
        color_dict = {   (2, 3): 'saddlebrown',
                         (1, 2): 'red',
                         (0.5, 1): 'lightsalmon',
                         (-0.5, 0.5): 'white',
                         (-1, -0.5): 'darkturquoise',
                         (-2, -1): 'teal',
                         (-3, -2): 'darkslategray',
                         (-4, -3): 'midnightblue',
                     }


    return color_dict

def mask_cube_by_basin(gdf, cube, PFAF_ID):
    import iris
    import geopandas as gpd
    import numpy as np
    import regionmask
    import numpy.ma as ma

    print('Masking cube by basin {}'.format(PFAF_ID))

    # To check its actually masking
    num_non_masked_points = ma.count(cube.data)
    print(f"Total number of non-masked grid points: {num_non_masked_points}")

    basin_geometry = gdf[gdf['PFAF_ID'] == PFAF_ID].geometry
    if len(basin_geometry) != 1:
        basin_geometry = basin_geometry.tolist()
    basin_mask = regionmask.Regions(basin_geometry)

    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points
    # Rasterises the regions to the lon, lat grid
    mask = basin_mask.mask(lon, lat)
    mask_cube = mask.to_iris()
    # Broadcast the mask to the shape of the input cube
    mask_array = np.broadcast_to(mask_cube.data, cube.shape)
    mask_array = ma.masked_invalid(mask_array)
    cube.data.mask = mask_array.mask

    num_non_masked_points = ma.count(cube.data)
    print(f"Total number of non-masked grid points after masking: {num_non_masked_points}")

    return cube

def get_riverbasin_PFAF_list():
    import geopandas as gpd
    basin_shpfile = '/data/users/jstacey/hydrobasins/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp'
    df = gpd.read_file(basin_shpfile)
    return df['PFAF_ID'].to_list()

def apply_basin_mask_to_cube(cube, PFAF_ID):
    import regionmask
    import geopandas as gpd
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

def get_pop_by_basin(yr_range, print_total=False):
    import iris
    import cf_units
    import pandas as pd
    from iris import coord_categorisation

    # load data
    fpath = '/data/users/jstacey/population_data/population_ssp2soc_0p5deg_annual_2006-2100.nc4'
    cube = iris.load_cube(fpath)

    # convert time units
    cube.coord("time").bounds = None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")

    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0)

    iris.coord_categorisation.add_year(cube, 'time')
    cube = cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))
    cube = cube.collapsed('time', iris.analysis.MEAN)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    if print_total:
        # to get total global population to test river basin pop sums up to this
        total_pop = cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM)
        print('TOTAL POP IS... {}'.format(total_pop.data))

    # Get list of all riverbasin IDs
    rb_list = get_riverbasin_PFAF_list()
    #rb_list = rb_list[0:1] #for testing

    # Create list of population total by basin averaged over the period
    pop_list = []
    for basin in rb_list:
        cube_basin = apply_basin_mask_to_cube(cube, basin)
        cube_basin = cube_basin.collapsed(['latitude', 'longitude'], iris.analysis.SUM)
        print('Pop count for basin {} is {}'.format(basin, cube_basin.data))
        pop_list.append(cube_basin.data.item())

    # make dataframe
    df = pd.DataFrame()
    df['PFAF_ID'] = rb_list
    #df = df.head(10)  # for testing
    df['pop_count'] = pop_list

    fname = 'table2_df_pop_by_basin-{}-{}.csv'.format(yr_range[0], yr_range[1])
    df.to_csv('/home/h06/jstacey/MSc/csv_files/{}'.format(fname))
    print('/home/h06/jstacey/MSc/csv_files/{} saved'.format(fname))

    return df

def calc_tot_pop():
    import pandas as pd
    import iris
    from iris import coord_categorisation
    import cf_units

    fpath = '/data/users/jstacey/population_data/population_ssp2soc_0p5deg_annual_2006-2100.nc4'
    cube = iris.load_cube(fpath)
    # convert time units
    cube.coord("time").bounds = None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")

    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0)
    # to get total global population to test river basin pop sums up to this
    iris.coord_categorisation.add_year(cube, 'time')
    cube = cube.extract(
        iris.Constraint(year=lambda cell: 2076 <= cell.point <= 2095))
    cube = cube.collapsed('time', iris.analysis.MEAN)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    total_pop_cube = cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM)

    df = pd.read_csv('/home/h06/jstacey/MSc/csv_files/table2_df_pop_by_basin-2076-2095.csv')
    total_pop_df = df['pop_count'].sum(skipna=True)
    print('TOTAL POP IS... {} from cube, {} from river basin df, diff is {}'.format(total_pop_cube.data, total_pop_df, total_pop_cube.data - total_pop_df))
    return total_pop_cube.data

def get_fut_pop_count_by_basin(PFAF_ID, in_millions=False):
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
    total_pop = 0
    for id in basin_list:
        pop_count = get_fut_pop_count_by_basin(id, in_millions=False)
        total_pop += pop_count
    return total_pop

def calc_basins_incr_decr(df, factor, threshold, table_dict):
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