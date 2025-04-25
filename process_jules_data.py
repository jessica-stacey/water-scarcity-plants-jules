'''
Author: Jessica Stacey
Paper: Future global  water scarcity partially alleviated by   vegetation responses to atmospheric CO2 and climate change (2025)

Initial loading and processing code for JULES output for various water cycle variables
which is then used by plotting functions Fig. 1-3
and to process water supply and demand in process_wsi.py for Figs 4-10
'''

from __future__ import (absolute_import, division, print_function)
import datetime as dt
import os


# -----------
# FUNCTIONS
# -----------

def load_and_process_jules_data(jules_out_fpath, experiment, var, year_range, output_dir=None):
    '''
    Load raw JULES output and process
    Args:
        jules_out_fpath: path to JULES output
        var: JULES output variable name
        year_range: list of start and end years
        output_dir: path to save output cube as .nc file
    Returns:
        cube: iris cube or saves .nc file
    '''

    import r6715_python_packages.share.jules as jules
    import iris
    from iris import cube
    from iris import coord_categorisation
    from iris.util import unify_time_units
    import numpy as np
    import numpy.ma as ma
    import cftime

    path_jobid_lut = {'u-bk886': jules_out_fpath}
    stream = 'monthly_profile'
    jobid = 'u-bk886'

    rcp_lut = {'historical': 'c20c',
               'rcp60': 'rcp6p0'}

    start = dt.datetime(year_range[0], 1, 1)
    end = dt.datetime(year_range[1], 1, 1)
    years = np.arange(start.year, end.year + 1)  # NB: JULES output in yearly streams # JS changed from end.year + 1

    if end <= dt.datetime(2006, 1, 1):
        rcp = 'historical'
    else:
        rcp = 'rcp60'

    # Assuming we test date using the logic start <= a_date < end
    hist = {'start': start, 'end': dt.datetime(2006, 1, 1)}
    futr = {'start': dt.datetime(2006, 1, 1), 'end': end}

    # Decide if we are straddling a historical / future range
    if start < hist['end'] and futr['start'] < end:
        # Then we have hist and futr data to merge together
        rcp_list = ['historical', rcp]
    else:
        rcp_list = [rcp]

    mod = 'HADGEM2-ES'
    print('   Loading ' + var.upper() + ': ' + mod)
    outcubelist = iris.cube.CubeList([])

    for onercp in rcp_list:
        path = path_jobid_lut[jobid] + mod + '/'
        for yr in years:
            infile = path + mod.lower() + '_' + rcp_lut[onercp] + '.' + stream + '.' + str(yr) + '.nc'
            if os.path.isfile(infile):
                try:
                    cube = jules.load_cube(infile, var)
                    if not cube.coord('latitude').has_bounds():
                        cube.coord('latitude').guess_bounds(bound_position=0.5)
                    if not cube.coord('longitude').has_bounds():
                        cube.coord('longitude').guess_bounds(bound_position=0.5)
                    cube.data = ma.masked_invalid(cube.data)
                    outcubelist.append(cube)
                    print(onercp, yr)
                except:
                    print('file for {} {} does not exist'.format(yr, onercp))
                    continue

    unify_time_units(outcubelist)
    cube = outcubelist.concatenate_cube()

    # Note about the time coordinate:
    #   By default, the stream is saved at the end of the aggregation period (month in this case)
    #   Meaning that the date point is the end of the period, and the bounds refer to the preceding month
    # make new time coord to avoid error
    dates = []
    years = np.arange(start.year, end.year + 1)
    for year in years:
        if year < 2100:
            for month in range(1, 13):
                dates.append(cftime.DatetimeGregorian(year, month, 1, 0, 0, 0, 0, has_year_zero=False))
        else:
            for month in range(1, 12):
                dates.append(cftime.DatetimeGregorian(year, month, 1, 0, 0, 0, 0, has_year_zero=False))

    tcoord = cube.coord("time")
    myu = tcoord.units
    tcoord.points = [myu.date2num(date) for date in dates]
    tcoord.convert_units("days since 1661-01-01 00:00:00")

    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0)  # assumes time is first dimension

    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')
    iris.coord_categorisation.add_season(cube, 'time')
    iris.coord_categorisation.add_season_year(cube, 'time')

    cube = cube.extract(
        iris.Constraint(time=lambda cell: start <= cell.point <= dt.datetime(2099, 12, 1)))  # to remove 2100

    cube.coord('time').bounds = None
    cube.coord('time').guess_bounds(0)
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds(0.5)
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds(0.5)

    if cube.units == 'kg m-2 s-1' and var in ['runoff', 'surf_roff', 'sub_surf_roff', 'et_stom_gb', 'esoil_gb',
                                              'fqw_gb', 'precip']:
        cube = iris.analysis.maths.multiply(cube, 86400)
        cube.units = 'mm/day'
    elif cube.units == 'K':
        cube.convert_units('celsius')
        print('Converted cube units Kelvin to Celsius')
    else:
        print('think about units!')

    # Mask all high values to remove error with gs - arctic not masking properly and giving values ~1e6!
    # This will probably take a while!
    if var == 'gs':
        gsdata = cube.data
        mask = ma.masked_where(gsdata > 1000, gsdata)
        cube.data = np.ma.masked_where(np.ma.getmask(mask), gsdata)

    std_name_dict = {'precip': 'precipitation_amount',
                     'runoff': 'runoff_amount',
                     'et_stom_gb': 'transpiration_amount'}
    try:
        cube.standard_name = std_name_dict[var]
    except:
        print('no standard name given')

    cube.attributes['title'] = '{}'.format(experiment)
    cube.long_name = '{}'.format(var)

    if output_dir is not None:
        fname = '{}_{}_mean_{}_{}-{}_monthly.nc'.format(mod, experiment, var, year_range[0], year_range[1])
        iris.save(cube, os.path.join(output_dir, fname))
        print('Cube saved to {}'.format(output_dir))
    else:
        print(cube)

    return cube


def main():
    '''
    Main function to load and process JULES output
    '''

    # -----------
    # ADJUSTABLE SETTINGS
    # -----------

    YEAR_RANGE = [1861, 2100]  # Full period: 1861 to 2100 - set up to work to FIRST month of end year

    # all variables - 'precip','runoff', 'rflow', 'ecan_gb', 'fqw_gb', 'et_stom_gb', 'lai', 'frac', 'surf_roff', 'sub_surf_roff', 'tstar_gb'
    VAR_LIST = ['lai_gb']
    EXPERIMENT_LIST = ['co2_triffid_fix']#, 'triffid_fix', 'co2_fix_noLUC']'all_noLUC',
    OUTPUT_DIR = '/data/users/jstacey/processed_jules_output/' # Make None if dont want to save cube

    FNAME_DICT = {'co2_triffid_fix': '/hpc/data/d01/jstacey/jules_output/ISIMIP_JS_output_CO2triffidfix/',
                  'triffid_fix': '/hpc/data/d01/jstacey/jules_output/ISIMIP_JS_output_triffidfix/',
                  'co2_fix_noLUC': '/hpc/data/d01/jstacey/jules_output/ISIMIP_JS_output_CO2fix_noLUC/',
                  'all_noLUC': '/hpc/data/d01/jstacey/jules_output/ISIMIP_JS_output_all_noLUC/',
                  }

    ####################
    ## EXECUTE FUNCTION
    ####################

    cube_dict = {}
    for var in VAR_LIST:
        for experiment in EXPERIMENT_LIST:
            cube = load_and_process_jules_data(jules_out_fpath=FNAME_DICT[experiment], experiment=experiment,
                                               var=var, year_range=YEAR_RANGE, output_dir=OUTPUT_DIR)

    return


if __name__ == '__main__':
    main()