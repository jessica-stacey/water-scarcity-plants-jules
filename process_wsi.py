'''
Author: Jessica Stacey
Paper: Future global  water scarcity partially alleviated by vegetation responses to atmospheric CO2 and climate change (2025)

Loading and processing data for water supply and demand
Water supply uses JULES output processed in process_jules_data.py
Water demand downloaded from ISIMIP2b data
'''

import os
import iris
from iris import cube
from iris import coord_categorisation

# ----------------------------
# FUNCTIONS
# -----------------------------

def extract_years(cube, yr_range):
    '''
    Extract years from cube
    Args:
        cube: iris cube
        yr_range: list of start and end years e.g., [start_year, end_year]

    Returns:
        cube: iris cube
    '''
    return cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))


def set_up_comparable_cubes(cube_in, use_time_format_from_cube):
    '''
    Set up cube with same time format as another cube
    Args:
        cube_in: iris cube
        use_time_format_from_cube: iris cube

    Returns: iris cube

    '''
    import numpy as np
    for vrb in ['time', 'latitude', 'longitude']:
        if cube_in.coord(vrb).has_bounds() == False:
            cube_in.coord(vrb).guess_bounds()

    time_coord = use_time_format_from_cube.coord('time')

    coord_list = [(time_coord, 0),
                  (cube_in.coord('latitude'), 1),
                  (cube_in.coord('longitude'), 2)]

    cube_data = cube_in.data
    cube_data = cube_data.astype(float)

    # Set up new cube inserting each coord
    cube_out = iris.cube.Cube(cube_data,
                              standard_name=cube_in.standard_name,
                              long_name=cube_in.long_name,
                              units=cube_in.units,
                              dim_coords_and_dims=coord_list)
    return cube_out


def convert_cube_units(cube, cube_units_out):
    '''
    Convert cube units
    Args:
        cube: iris cube must have units of 'kg m-2 s-1' or 'mm/day'
        cube_units_out: 'm**3/day', 'm**3/sec'

    Returns: iris cube

    '''
    # water supply = runoff / rflow in mm/day / kg m-2 s-1
    # water demand in m^3/day
    # to convert to same units
    # mm/day = 1 litre per m2
    # m^3/day = 1000 litres
    if (cube.units == 'kg m-2 s-1' and cube_units_out == 'm**3/day'):
        # convert to mm/day
        cube = iris.analysis.maths.multiply(cube, 86400)
        cube.units = 'mm/day'
        wghts = iris.analysis.cartography.area_weights(cube, normalize=False)  # in m**2
        wghts_cube = cube.copy(data=wghts)
        wghts_cube.units = "m**2"
        cube = cube * wghts_cube  # units are now a fraction of m**3.s**-1
        cube.convert_units("m**3/day")
    elif (cube.units == 'kg m-2 s-1' and cube_units_out == 'm**3/sec'):
        cube.units = 'mm/sec'
        wghts = iris.analysis.cartography.area_weights(cube, normalize=False)  # in m**2
        wghts_cube = cube.copy(data=wghts)
        wghts_cube.units = "m**2"
        cube = cube * wghts_cube  # units are now a fraction of m**3.s**-1
        cube.convert_units("m**3.s**-1")
    elif (cube.units == 'mm/day' and cube_units_out == 'm**3/day'):
        wghts = iris.analysis.cartography.area_weights(cube, normalize=False)  # in m**2
        wghts_cube = cube.copy(data=wghts)
        wghts_cube.units = "m**2"
        cube = cube * wghts_cube  # units are now a fraction of m**3.s**-1
        cube.convert_units("m**3/day")
    else:
        raise ValueError('Convert_units() does not currently work for the given cube units in and out')

    return cube


def get_water_supply_cube(climate_model, experiment, water_supply_dir, var='runoff', yr_range=[2006, 2099]):
    '''
    Load and process water supply cube
    Args:
        climate_model: str 'HADGEM2-ES'
        experiment: 'all_noLUC', 'triffid_fix', 'co2_triffid_fix', 'co2_fix_noLUC'
        water_supply_dir: directory to find JULES output files created in process_jules_data.py
        var: variable to use as proxy for water supply, e.g., 'runoff' or 'rflow'
        yr_range: [start_year, end_year]

    Returns: iris cube

    '''
    fname = os.path.join(water_supply_dir,
                         '{}_{}_mean_{}_1861-2100_monthly.nc'.format(climate_model, experiment, var))
    water_supply_cube = iris.load_cube(fname)

    water_supply_cube = extract_years(water_supply_cube, yr_range=yr_range)
    water_supply_cube = convert_cube_units(water_supply_cube, 'm**3/day')
    formatted_cube = set_up_comparable_cubes(water_supply_cube, use_time_format_from_cube=water_supply_cube)
    formatted_cube.long_name = 'water supply'
    return formatted_cube


def determine_year_from_isimip_demand_time(coord, month):
    '''
    Determine year from ISIMIP2 demand cube time coordinate
    Args:
        coord: time coord as named in cube
        month: integer

    Returns: year as integer

    '''
    import math
    year = math.floor((month / 12) + 1661)
    return year

def get_isimip_demand_cube(climate_model, hydro_model, water_demand_sources, water_demand_dir, water_supply_cube,
                           yr_range=[2006, 2099]):
    '''
    Load and process ISIMIP2 demand data
    Args:
        climate_model: str, 'HADGEM2-ES'
        hydro_model: str, 'H08'
        water_demand_sources: list, ['industrial', 'domestic', 'irrigation']
        water_demand_dir: directory path where demand data saved
        water_supply_cube: directory path where supply data saved (from process_jules_data.py)
        yr_range: [start_year, end_year]

    Returns: iris cube

    '''
    # load and sum up ISIMIP demand data
    demand_temp_list = []
    for source_type in water_demand_sources:
        if source_type == 'domestic':
            fname = '{}_{}_ewembi_rcp60_rcp60soc_co2_adomww_landonly_monthly_2006_2099.nc4'.format(
                hydro_model.lower(), climate_model.lower())
        elif source_type == 'industrial':
            fname = '{}_{}_ewembi_rcp60_rcp60soc_co2_amanww_landonly_monthly_2006_2099.nc4'.format(
                hydro_model.lower(), climate_model.lower())
        elif source_type == 'irrigation':
            fname = '{}_{}_ewembi_rcp60_rcp60soc_co2_pirrww_landonly_monthly_2006_2099.nc4'.format(
                hydro_model.lower(), climate_model.lower())
        else:
            print('water_demand_sources should include in list domestic, industrial or irrigation')

        cube = iris.load_cube(os.path.join(water_demand_dir, fname))
        demand_temp_list.append(cube)

    tot_cube = sum(demand_temp_list)

    # make grid same as supply cube
    tot_cube = tot_cube.extract(iris.Constraint(latitude=lambda cell: -55.75 <= cell.point <= 83.75)) # to make consistent with supply
    tot_cube = iris.util.reverse(tot_cube, 'latitude')
    for vrb in ['time', 'latitude', 'longitude']:
        if tot_cube.coord(vrb).has_bounds() == False:
            tot_cube.coord(vrb).guess_bounds()

    # original time in 360-calendar and months since 1661, so manually make new year coord
    iris.coord_categorisation.add_categorised_coord(tot_cube, 'year', 'time',
                                                    determine_year_from_isimip_demand_time)
    tot_cube.coord('year').units = 'year'

    tot_cube = extract_years(tot_cube, yr_range=yr_range)
    tot_cube = convert_cube_units(tot_cube, 'm**3/day')
    formatted_cube = set_up_comparable_cubes(tot_cube, use_time_format_from_cube=water_supply_cube)
    formatted_cube.long_name = 'water demand'

    return formatted_cube


def mask_supply_demand_cubes_where_demand_negative(water_supply_dict, demand_cube):
    '''
    Mask supply and demand cubes where demand is negative in H08 output data (invesitgating this with ISIMIP)
    Args:
        water_supply_dict: dictionary of cubes by experiment
        demand_cube: iris cube of processed demand data

    Returns: dictionary of cubes by experiment, iris cube

    '''

    import numpy as np
    demand_cube.data = np.ma.masked_less(demand_cube.data, 0)

    for experiment in water_supply_dict.keys():
        water_supply_dict[experiment].data = np.ma.masked_where(demand_cube.data.mask,
                                                                water_supply_dict[experiment].data)

    return water_supply_dict, demand_cube

def main():
    '''
    Main function to load and process water supply and demand data
    '''
    #
    ########################
    # ADJUSTABLE SETTINGS
    #######################
    EXPERIMENT_LIST = ['all_noLUC']#, 'triffid_fix', 'co2_triffid_fix', 'co2_fix_noLUC']
    YR_RANGE = [2006, 2100]
    CLIMATE_MODEL = 'HADGEM2-ES'
    HYDRO_MODEL = 'h08'
    WATER_SUPPLY_VAR = 'runoff'
    WATER_DEMAND_SOURCES = ['industrial', 'domestic', 'irrigation']
    SSP = 'ssp2'

    # Directories
    WATER_SUPPLY_INPUT_DIR = '/data/users/jstacey/processed_jules_output/'
    WATER_DEMAND_INPUT_DIR = '/data/users/jstacey/water_demand/ISIMIP2/'
    OUTPUT_DIR = '/data/users/jstacey/water_demand/ISIMIP2/'

    ##############################################################
    # Load, process and save water supply and demand cubes
    ###########################################################

    water_supply_dict = {}
    for experiment in EXPERIMENT_LIST:
        print('Loading water supply cube for {}, experiment {}'.format(CLIMATE_MODEL, experiment))
        supply_cube = get_water_supply_cube(climate_model=CLIMATE_MODEL, experiment=experiment,
                                            water_supply_dir=WATER_SUPPLY_INPUT_DIR,
                                            var=WATER_SUPPLY_VAR, yr_range=YR_RANGE)
        water_supply_dict[experiment] = supply_cube


    water_demand_dict = {}
    print('Loading water demand cube')
    demand_cube = get_isimip_demand_cube(CLIMATE_MODEL, HYDRO_MODEL, water_demand_sources=WATER_DEMAND_SOURCES,
                                         water_demand_dir=WATER_DEMAND_INPUT_DIR,
                                         # to copy format - doesnt matter which experiment:
                                         water_supply_cube=water_supply_dict['all_noLUC'],
                                         yr_range=YR_RANGE)

    water_supply_dict, demand_cube = mask_supply_demand_cubes_where_demand_negative(water_supply_dict, demand_cube)
    fname = 'demand_{}_{}_ssp2_{}-{}_monthly-neg_demand_masked.nc'.format(CLIMATE_MODEL, HYDRO_MODEL, YR_RANGE[0], YR_RANGE[1])
    iris.save(demand_cube, os.path.join(OUTPUT_DIR, fname))
    print('demand cube saved to {}'.format(OUTPUT_DIR))

    for experiment in EXPERIMENT_LIST:
        supply_cube = water_supply_dict[experiment]
        fname = 'supply_{}_{}_{}-{}_{}_monthly-neg_demand_masked.nc'.format(CLIMATE_MODEL, experiment, YR_RANGE[0], YR_RANGE[1],
                                                          WATER_SUPPLY_VAR)
        iris.save(supply_cube, os.path.join(OUTPUT_DIR, fname))
    print('supply cubes saved to {}'.format(OUTPUT_DIR))

    return

if __name__ == '__main__':
    main()