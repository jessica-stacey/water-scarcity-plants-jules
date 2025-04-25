# Main file for processing and plotting paper figures
# Jessica Stacey
import iris
import iris.plot as iplt
import numpy as np
import matplotlib.pyplot as plt
import common_functions_paper as common
plt.rcParams['font.family'] = 'Arial'

def preprocess_fig1a_global_watercycle_timeseries(var_list):
    '''
    Preprocesses the data for input variable timeseries plots in Fig 1
    Args:
        var_list: list of variables to be plotted

    Returns: dictionary of cubes for each variable

    '''

    from iris import analysis
    from iris.analysis import cartography

    cube_dict = {}

    for var in var_list:
        print('Loading and formatting {} cube'.format(var))
        for experiment in ['co2_triffid_fix']:
            if var == 'co2_mmr':
                cube = common.load_co2_mmr()
                cube = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
            else:
                fname = '/data/users/jstacey/processed_jules_output/HADGEM2-ES_co2_triffid_fix_mean_{}_1861-2100_monthly.nc'.format(var)
                cube = iris.load_cube(fname)
                cube = cube.aggregated_by('year', iris.analysis.MEAN)
                cube = cube.extract(iris.Constraint(year=lambda cell: 1900 <= cell.point <= 2100))

                for vrb in ['latitude', 'longitude', 'time']:
                    if cube.coord(vrb).has_bounds() == False:
                        cube.coord(vrb).guess_bounds()

                cube = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN,
                                      weights=iris.analysis.cartography.area_weights(cube))

                cube = cube.rolling_window('year', iris.analysis.MEAN, 5)

            cube_dict[var] = cube

    return cube_dict

def plot_fig1a_global_input_vars_timeseries(plot_dir, fontsize=18):
    '''
    Plots the timeseries of the input data: temperature, precipitation and atmospheric CO2
    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plots as .png

    '''

    var_list = [ 't1p5m_gb', 'precip', 'co2_mmr']
    cube_dict = preprocess_fig1a_global_watercycle_timeseries(var_list)

    print('Plotting fig 1: timeseries of water cycle variables by experiment')
    fig = plt.figure(figsize=(17, 3))
    fig.subplots_adjust(hspace=0.02, wspace=0.25, left=0.075, right=0.8)
    panel = 0
    for var in var_list:
        var_title = common.get_var_title(var)
        print('Plotting var {}'.format(var_title))

        panel = panel + 1
        ax1 = fig.add_subplot(1, len(var_list), panel)

        for experiment in ['co2_triffid_fix']:
            cube = cube_dict[var]
            if var == 'co2_mmr':
                x_vals = cube.coord('time').points
            else:
                x_vals = cube.coord('year').points
            y_vals = cube.data
            label = common.get_experiment_label(experiment)
            color = common.get_experiment_color_dict(experiment)
            plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)

            unit_title = common.get_unit_title(var)
            ax1.set_title('{} ({})'.format(var_title, unit_title), fontsize=fontsize + 2)

            ax1.tick_params(labelsize=fontsize)
            plt.xticks(rotation=45)
            plt.locator_params(nbins=5, axis='x')

            if panel == 1:
                ax1.set_ylabel('Global annual mean', fontsize=fontsize)
            if panel == 3:
                ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=fontsize + 2, fancybox=True,
                           shadow=True)

    fname = 'fig1a_global_mean_timeseries_input_vars.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return


def preprocess_fig1b_fig3_global_watercycle_maps(var_list, year_range_dict):
    '''
    Preprocesses the data for the global water cycle maps in Fig 3 and 4, calculating the change between future and historical periods
    Args:
        var_list: list of variable names
        year_range_dict: dictionary of list of years for historical and future periods,
            e.g., dict['hist'][start year, end year] and dict['fut'][start, end]

    Returns: dictionary of cubes for each variable and experiment

    '''
    from iris import analysis
    cube_dict = {}

    for var in var_list:
        print('Loading and formatting {} cube'.format(var))
        for experiment in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:
            fname = '/data/users/jstacey/processed_jules_output/HADGEM2-ES_{}_mean_{}_1861-2100_monthly.nc'.format(
                experiment, var)
            cube = iris.load_cube(fname)
            cube = cube.extract(iris.Constraint(year=lambda cell: 2005 <= cell.point <= 2100))
            #cube = cube.aggregated_by('year', iris.analysis.MEAN)

            for vrb in ['latitude', 'longitude', 'time']:
                if cube.coord(vrb).has_bounds() == False:
                    cube.coord(vrb).guess_bounds()
            # radius_of_earth = DEFAULT_SPHERICAL_EARTH_RADIUS

            if cube.units == 'kg m-2 s-1':
                print('Cube units for {} in kg m-2 s-1 - converting to mm/day'.format(var))
                cube = iris.analysis.maths.multiply(cube, 86400)
                cube.units = 'mm/day'

            cube_dict[var, experiment] = cube

    change_dict = {}
    for var in var_list:
        for experiment in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:
            cube = cube_dict[var, experiment]
            cube_hist = cube.extract(iris.Constraint(year=lambda cell: year_range_dict['hist'][0] <= cell.point <= year_range_dict['hist'][1]))
            cube_fut = cube.extract(iris.Constraint(year=lambda cell: year_range_dict['fut'][0] <= cell.point <= year_range_dict['fut'][1]))
            cube_hist = cube_hist.collapsed('time', iris.analysis.MEAN)
            cube_fut = cube_fut.collapsed('time', iris.analysis.MEAN)
            change_cube = cube_fut - cube_hist
            change_dict[var, experiment] = change_cube

    return change_dict

def plot_fig1b_global_change_temp_ppn_map(plot_dir, fontsize=18):
    '''
    Plots the global maps of change in temperature and precipitation (input variables) between future and historical periods
    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plots as .png

    '''
    import cartopy.crs as ccrs

    print('Calculating change between future and historical periods for both experiments and enter into dict')
    year_range_dict = {'hist': [2006, 2025],
                       'fut': [2076, 2095]}

    var_list = ['t1p5m_gb', 'precip']

    change_dict = preprocess_fig1b_fig3_global_watercycle_maps(var_list, year_range_dict=year_range_dict)

    fig = plt.figure(figsize=(10, 3))  # (16, 6.5))
    fig.subplots_adjust(hspace=0.02, wspace=0.02, bottom=0.1, top=0.95, left=0.1, right=0.99)

    panel = 0
    contr_factor_list = ['CLIM']  # , 'ALL', 'PLANT_PHYS', 'VEG_DIST', 'PLANT_PHYS_VEG'] #'STOM',
    letters = 'abcdefghijklmnopqrstuvwxyz'  # for subplot letter labels

    for var in var_list:
        print('Plotting maps of {} change'.format(var))
        panel += 1
        ax = fig.add_subplot(1, len(var_list), panel)

        if var == 'precip':
            levels = np.linspace(-0.5, 0.5, 11)
            cmap = 'BrBG'
            extend = 'both'
        elif var == 't1p5m_gb':
            levels = np.linspace(0, 6, 13)
            cmap = 'YlOrRd'
            extend = 'max'

        cube_to_plot = change_dict[var, 'co2_triffid_fix']

        cf = iplt.contourf(cube_to_plot, levels=levels, extend=extend, cmap=plt.get_cmap(cmap))

        var_title = common.get_var_title(var)
        contr_factor_name = common.get_contr_factor_label_maps('CLIM')
        plt.title('{}'.format(var_title), fontsize=fontsize)

        if panel == 1:
            plt.text(-0.01, 0.5, contr_factor_name, fontsize=fontsize, transform=plt.gcf().transFigure,
                     rotation='horizontal', bbox=dict(facecolor='lightgrey', alpha=0.3))

        plt.gca().coastlines()

        cbar_unit_title = common.get_unit_title(var)
        # cbar_axes = fig.add_axes([0.2, 0.05, 0.6, 0.05])
        # common.add_cbar(fig, cf, cbar_unit_title, var=var, font_size=18, cbar_axes_loc=cbar_axes)
        cbar = plt.colorbar(cf, pad=0.1, orientation='horizontal')
        cbar.set_label('{}'.format(cbar_unit_title), fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    fname = 'fig1b_map_mean_change_temp_ppn.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return

def preprocess_fig2_global_watercycle_timeseries(var_list):
    '''
    Preprocesses the data for the global water cycle timeseries plots in Fig 2
    Args:
        var_list: list of variable names

    Returns: dictionary of cubes for each variable and experiment

    '''
    from iris import analysis
    from iris.analysis import cartography

    cube_dict = {}

    for var in var_list:
        print('Loading and formatting {} cube'.format(var))
        for experiment in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:
            fname = '/data/users/jstacey/processed_jules_output/HADGEM2-ES_{}_mean_{}_1861-2100_monthly.nc'.format(
                experiment, var)
            cube = iris.load_cube(fname)
            cube = cube.aggregated_by('year', iris.analysis.MEAN)
            cube = cube.extract(iris.Constraint(year=lambda cell: 1900 <= cell.point <= 2100))

            for vrb in ['latitude', 'longitude', 'time']:
                if cube.coord(vrb).has_bounds() == False:
                    cube.coord(vrb).guess_bounds()

            cube = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN,
                                  weights=iris.analysis.cartography.area_weights(cube))

            cube = cube.rolling_window('year', iris.analysis.MEAN, 5)
            cube_dict[var, experiment] = cube

    return cube_dict




def plot_fig2_global_multivar_timeseries(plot_dir, var_list_no, fontsize=16):
    '''
    Plots the timeseries of the water cycle variables for each experiment
    - these are then joined together in powerpoint!
    Args:
        var_list_no: 1 (top row of variables) or 2 (bottom row of variables) in Fig. 2
        fontsize: int, fontsize for labels

    Returns: Saves plots as .png
    '''

    if var_list_no == 1:
        var_list = ['runoff', 'surf_roff', 'sub_surf_roff', 'smc_tot']
        letters = 'abcdefgh'
    elif var_list_no == 2:
        var_list = ['et_stom_gb', 'gs', 'lai_gb']
        letters = 'ijklmn'
    else:
        print('var_list_no. should be assigned to number 1 or 2')

    cube_dict = preprocess_fig2_global_watercycle_timeseries(var_list)
    contr_factor_dict = common.calc_contr_factors_timeseries(cube_dict, var_list, calc_rel_diff=True)

    print('Plotting fig 1: timeseries of water cycle variables by experiment')
    fig = plt.figure(figsize=(17.5, 6))


    if len(var_list) == 4:
        fig.subplots_adjust(hspace=0.02, wspace=0.25, left=0.075, right=0.99)
    elif len(var_list) == 3: # allow room for legend
        fig.subplots_adjust(hspace=0.02, wspace=0.25, left=0.075, right=0.75)

    panel = 0
    for var in var_list:
        panel = panel + 1
        ax1 = fig.add_subplot(2, len(var_list), panel)
        var_title = common.get_var_title(var)
        print('Plotting var {}'.format(var_title))
        for experiment in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:

            cube = cube_dict[var, experiment]
            label = common.get_experiment_label(experiment)
            x_vals = cube.coord('year').points
            y_vals = cube.data
            color = common.get_experiment_color_dict(experiment)
            plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)

            unit_title = common.get_unit_title(var)
            if var == 'lai_gb':
                ax1.set_title('Leaf Area Index', fontsize=fontsize+2)  # no units
                ax1.set_ylim([1.8, 2.5])
            else:
                ax1.set_title('{} ({})'.format(var_title, unit_title), fontsize=fontsize+2)

            if var == 'gs': # edit ticks s othey are only to 4 decimal places
                plt.yticks(ticks=[0.0025, 0.003, 0.0035], labels=[0.0025, 0.0030, 0.0035])
                #ax1.set_ylim([0.00245, 0.0037])

            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax1.tick_params(labelsize=fontsize)

            if panel == 1:
                ax1.set_ylabel('Global annual mean', fontsize=fontsize)
            if var_list_no == 2 and panel == 3:
                ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),  ncol=1, fontsize=fontsize, fancybox=True, shadow=True)

            if var_list_no == 2 or var=='smc_tot':
                plt.text(0.02, 0.04, letters[panel - 1], transform=ax1.transAxes, fontsize=20, fontweight='bold',
                     va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))
            else:
                plt.text(0.02, 0.96, letters[panel - 1], transform=ax1.transAxes, fontsize=20, fontweight='bold',
                         va='top', ha='left', bbox=dict(facecolor='white', alpha=0.3))


    for var in var_list:
        panel = panel + 1
        ax2 = fig.add_subplot(2, len(var_list), panel)
        ax2.axhline(y=0, color='dimgray', linestyle='--')
        for contr_factor in ['VEG_DIST', 'STOM', 'PLANT_PHYS',  'PLANT_PHYS_VEG']:
            cube = contr_factor_dict[var, contr_factor]
            x_vals = cube.coord('year').points
            y_vals = cube.data
            label = common.get_contr_factor_label(contr_factor)
            color = common.get_contr_factor_color_dict(contr_factor)
            if contr_factor == 'PLANT_PHYS_VEG':
                plt.plot(x_vals, y_vals, color=color, label=label, linewidth=2.0) # thicker line
            else:
                plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)

        ax2.tick_params(labelsize=fontsize)
        plt.xticks(rotation=45)
        plt.locator_params(nbins=5, axis='x')

        if panel == (len(var_list)+1):
            ax2.set_ylabel('% difference', fontsize=fontsize)
        if var_list_no == 2 and panel == 6:
            ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),  ncol=1, fontsize=fontsize, fancybox=True, shadow=True)

        if var_list_no == 2:
            plt.text(0.02, 0.04, letters[panel - 1], transform=ax2.transAxes, fontsize=20, fontweight='bold',
                     va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))
        else:
            plt.text(0.02, 0.96, letters[panel - 1], transform=ax2.transAxes, fontsize=20, fontweight='bold',
                     va='top', ha='left', bbox=dict(facecolor='white', alpha=0.3))

    fname = 'fig2_global_mean_timeseries_varlist-{}.png'.format(var_list_no)
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return


def plot_fig3_global_change_multivar_factors_map(plot_dir):
    '''
    Plots the global maps of change in water cycle variables between future and historical periods

    Returns: Saves plots as .png

    '''
    var_list = ['runoff', 'smc_tot', 'et_stom_gb',  'lai_gb']
    year_range_dict = {'hist': [2006, 2025],
                       'fut': [2076, 2095]}

    change_dict = preprocess_fig1b_fig3_global_watercycle_maps(var_list, year_range_dict=year_range_dict)
    contr_factor_dict = common.calc_contr_factors_map(cube_dict=change_dict, var_list=var_list)

    fig = plt.figure(figsize=(16, 8))#(16, 6.5))
    fig.subplots_adjust(hspace=0.02, wspace=0.02, bottom=0.1, top=0.95, left=0.1, right=0.99)

    panel = 0
    contr_factor_list = ['CLIM', 'ALL',  'PLANT_PHYS', 'VEG_DIST', 'PLANT_PHYS_VEG'] #'STOM',
    letters = 'abcdefghijklmnopqrstuvwxyz' # for subplot letter labels

    for row, contr_factor in enumerate(contr_factor_list):
        for col, var in enumerate(var_list):
            print('Plotting maps of {} change for {}'.format(var, contr_factor))
            panel += 1
            ax = fig.add_subplot(len(contr_factor_list), len(var_list), panel)

            if var == 'gs':
                levels = np.linspace(-0.002, 0.002, 11)
            elif var == 'lai_gb':
                levels = np.linspace(-1, 1, 11)
            elif var == 'smc_tot':
                levels = np.linspace(-100, 100, 11)
            else:
                levels = np.linspace(-0.5, 0.5, 11)

            var_title = common.get_var_title(var)

            if var == 'et_stom_gb':
                cmap = 'BrBG_r'
            elif var in ['lai_gb', 'gs']:
                cmap = 'PiYG'
            else:
                cmap = 'BrBG'

            if contr_factor=='ALL':
                cube_to_plot = change_dict[var, 'all_noLUC']
            else:
                cube_to_plot = contr_factor_dict[var, contr_factor]
            cf = iplt.contourf(cube_to_plot, levels=levels, extend="both", cmap=plt.get_cmap(cmap))

            if row == 0:  # plot titles on top row only
                plt.title('{}'.format(var_title), fontsize=20)

            if col == 0:  # Add labels to the left-hand column
                contr_factor_name = common.get_contr_factor_label_maps(contr_factor)
                facecolor_dict = {'CLIM': 'lightgrey', 'ALL': 'lightgrey', 'STOM': 'yellowgreen',
                                  'PLANT_PHYS': 'yellowgreen', 'VEG_DIST': 'yellowgreen', 'PLANT_PHYS_VEG': 'yellowgreen'}
                if contr_factor == 'ALL':
                    plt.text(-0.48, 0.3, contr_factor_name, fontsize=20, transform=ax.transAxes,
                             rotation='horizontal', bbox=dict(facecolor=facecolor_dict[contr_factor], alpha=0.3))
                elif contr_factor == 'PLANT_PHYS_VEG':
                    plt.text(-0.48, 0.1, contr_factor_name, fontsize=20, transform=ax.transAxes,
                             rotation='horizontal', bbox=dict(facecolor=facecolor_dict[contr_factor], alpha=0.3))
                else:
                    plt.text(-0.48, 0.4, contr_factor_name, fontsize=20, transform=ax.transAxes,
                         rotation='horizontal', bbox=dict(facecolor=facecolor_dict[contr_factor], alpha=0.3))
            # plot letter labels
            plt.text(0.02, 0.04, letters[panel - 1], transform=ax.transAxes, fontsize=20, fontweight='bold',
                     va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))

            plt.gca().coastlines()

            if row == len(contr_factor_list) - 1:
                # Plot legend on bottom row only
                cbar_axes_loc = common.position_cbar_colno_dependant(len(var_list), col)
                cbar_unit_title = common.get_unit_title(var)
                common.add_cbar(fig, cf, cbar_unit_title, var=var, cbar_axes_loc=cbar_axes_loc)

    fname = 'fig3_map_mean_change_contr_factors.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return


def preprocess_fig4_global_demand_supply_wsi_timeseries():
    '''
    Preprocesses the data for the global water cycle timeseries plots in Fig 5
    Returns: dictionary of cubes for each variable and experiment

    '''
    import iris
    from iris import coord_categorisation
    import common_functions_paper as common
    import numpy as np

    cube_out_dict = {}

    print('Processing demand cube')
    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly-neg_demand_masked.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    monthly_demand_cube = monthly_demand_cube.extract(iris.Constraint(year=lambda cell: 2006 <= cell.point <= 2099))

    num_values_above_threshold = np.sum(monthly_demand_cube.data < 0)
    total_values = monthly_demand_cube.data.size
    percentage_changed_0 = (num_values_above_threshold / total_values) * 100
    print(f"Number of negative values: {total_values:.2f} {percentage_changed_0:.2f}%")

    # For demand output fig5a
    ann_demand_cube = monthly_demand_cube.aggregated_by('year', iris.analysis.MEDIAN)
    global_ann_demand_cube = ann_demand_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEDIAN)
    cube_out_dict['demand', 'co2_triffid_fix'] = global_ann_demand_cube

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Processing supply cube')
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly-neg_demand_masked.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(iris.Constraint(year=lambda cell: 2006 <= cell.point <= 2099))

        num_values_above_threshold = np.sum(monthly_supply_cube.data < 0)
        total_values = monthly_supply_cube.data.size
        percentage_changed_0 = (num_values_above_threshold / total_values) * 100
        print(f"Number of negative values: {total_values:.2f} {percentage_changed_0:.2f}%")

        # For supply output Fig 4b
        ann_supply_cube = monthly_supply_cube.aggregated_by('year', iris.analysis.MEDIAN)
        global_ann_supply_cube = ann_supply_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEDIAN)
        cube_out_dict['supply', experiment] = global_ann_supply_cube

        print('Processing WSI cube')
        monthly_wsi_cube = monthly_demand_cube / monthly_supply_cube
        ann_wsi_cube = monthly_wsi_cube.aggregated_by('year', iris.analysis.MEDIAN)
        global_ann_wsi_cube = ann_wsi_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEDIAN)
        cube_out_dict['wsi', experiment] = global_ann_wsi_cube

    return cube_out_dict


def plot_fig4_global_demand_supply_wsi_timeseries(plot_dir, fontsize=14):
    '''
    Plots the timeseries of the water cycle variables for each experiment
    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plots as .png

    '''
    import common_functions_paper as common
    import matplotlib.pyplot as plt
    import iris

    cube_dict = preprocess_fig4_global_demand_supply_wsi_timeseries()
    contr_factor_dict = common.calc_contr_factors_timeseries(cube_dict, ['supply', 'wsi'], calc_rel_diff=True)

    print('Plotting fig 5: timeseries of water supply, demand, WSI by experiment')
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.3, top=0.95, bottom=0.01, left=0.075, right=0.8)

    letters = 'abcde'

    # Plot supply and WSI seperately, as also have additional plot showing contr factors
    for col, var in enumerate(['demand', 'supply', 'wsi']):  # plot line for each experiment
        var_title = common.get_var_title(var)
        ax1 = fig.add_subplot(2, 3, col + 1)
        for experiment in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:
            cube = cube_dict[var, experiment]
            cube = cube.rolling_window('year', iris.analysis.MEAN, 5)
            # if var=='supply':
            #    cube = cube/1000000 # to fit y-labels in - added x10^6 to title
            label = common.get_experiment_label(experiment)
            x_vals = cube.coord('year').points
            y_vals = cube.data
            color = common.get_experiment_color_dict(experiment)
            plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)
            if var == 'demand':
                ax1.set_ylabel('Global annual median', fontsize=fontsize)
            if var == 'wsi':
                ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=fontsize, fancybox=True,
                           shadow=True)
            plt.text(0.022, 0.96, letters[col], transform=ax1.transAxes, fontsize=fontsize, fontweight='bold',
                     va='top', ha='left', bbox=dict(facecolor='white', alpha=0.3))  # (horizontal, vertical, ...)
            if var == 'demand':  # only need to plot the one experiment as same for all
                break
        ax1.set_title(var_title, fontsize=fontsize + 2)

        ax1.tick_params(labelsize=fontsize-1.5)

    for col, var in enumerate(['supply', 'wsi']):
        ax2 = fig.add_subplot(2, 3, col + 5)
        for contr_factor in ['VEG_DIST', 'STOM', 'PLANT_PHYS', 'PLANT_PHYS_VEG']:
            cube = contr_factor_dict[var, contr_factor]
            cube = cube.rolling_window('year', iris.analysis.MEAN, 5)
            label = common.get_contr_factor_label(contr_factor)
            x_vals = cube.coord('year').points
            y_vals = cube.data
            color = common.get_contr_factor_color_dict(contr_factor)
            if contr_factor == 'PLANT_PHYS_VEG':
                plt.plot(x_vals, y_vals, color=color, label=label, linewidth=2)  # thicker line
            else:
                plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)

            ax2.tick_params(labelsize=fontsize-1.5)
            ax2.axhline(y=0, color='dimgray', linestyle='--')
            plt.locator_params(nbins=5, axis='x')
            if var == 'supply':
                ax2.set_ylabel('% difference', fontsize=fontsize)
            if var == 'wsi':
                ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=fontsize, fancybox=True,
                           shadow=True)
                ax2.set_ylim([-20, 0])
            plt.text(0.022, 0.02, letters[col + 3], transform=ax2.transAxes, fontsize=fontsize, fontweight='bold',
                     va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))

    fname = 'fig4_demand_supply_wsi_contr_factors_timeseries.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))
    return

def plot_fig4_as_key_fig(plot_dir, fontsize=14):
    '''
    Plots the key figure edited from Fig 5, showing the timeseries of WSI and the relative difference in WSI when all plant responses are included
    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plot as .png

    '''
    import common_functions_paper as common
    import matplotlib.pyplot as plt
    import iris

    cube_dict = preprocess_fig4_global_demand_supply_wsi_timeseries()
    contr_factor_dict = common.calc_contr_factors_timeseries(cube_dict, ['supply', 'wsi'], calc_rel_diff=True)

    print('Plotting fig 1: timeseries of water cycle variables by experiment')
    fig = plt.figure(figsize=(7, 8))
    fig.subplots_adjust(hspace=0.3, top=0.9, bottom=0.1, left=0.075, right=0.8)

    # Plot supply and WSI seperately, as also have additional plot showing contr factors
    ax1 = fig.add_subplot(2, 1, 1)
    for experiment in ['co2_triffid_fix', 'all_noLUC']:
        cube = cube_dict['wsi', experiment]
        cube = cube.rolling_window('year', iris.analysis.MEAN, 5)
        if experiment == 'co2_triffid_fix':
            label = 'Fixed plant responses'
        elif experiment == 'all_noLUC':
            label = 'Dynamic plant responses'
        x_vals = cube.coord('year').points
        y_vals = cube.data
        color = common.get_experiment_color_dict(experiment)
        plt.plot(x_vals, y_vals, color=color, label=label, linewidth=1.5)
    ax1.legend(loc='best', ncol=1, fontsize=fontsize)#, fancybox=True, shadow=True)
    ax1.set_ylabel('Global annual median', fontsize=fontsize)
    ax1.set_title('Water Scarcity Index', fontsize=fontsize + 2)
    ax1.tick_params(labelsize=fontsize)

    ax2 = fig.add_subplot(2, 1, 2)
    cube = contr_factor_dict['wsi', 'PLANT_PHYS_VEG']
    cube = cube.rolling_window('year', iris.analysis.MEAN, 5)
    x_vals = cube.coord('year').points
    y_vals = cube.data
    color = common.get_contr_factor_color_dict('PLANT_PHYS_VEG')
    label = 'Dynamic vs. fixed plant responses'
    plt.plot(x_vals, y_vals, color=color, label=label, linewidth=2)  # thicker line
    ax2.tick_params(labelsize=fontsize)
    ax2.axhline(y=0, color='dimgray', linestyle='--')
    plt.locator_params(nbins=5, axis='x')
    ax2.set_ylim([-20, 0])
    ax2.set_title('Relative difference when all plant responses included', fontsize=fontsize + 2)
    ax2.set_ylabel('%', fontsize=fontsize)
    ax2.set_xlabel('Year', fontsize=fontsize)

    ax2.legend(loc='best', ncol=1, fontsize=fontsize)

    fname = 'fig4_key_fig_wsi_contr_factors_timeseries.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return


def preprocess_fig5_fig6_demand_supply_wsi_present_and_change_map():
    '''
    Preprocesses the data for the global water cycle maps in Fig 6, calculating the change between future and historical periods
    Returns: dictionary of cubes for each variable and experiment

    '''
    import iris
    from iris import coord_categorisation
    import common_functions_paper as common

    temp_dict = {}

    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    for period in ['present', 'fut']:
        yr_range = [2006, 2025] if period == 'present' else [2076, 2095]
        demand_cube_ext = monthly_demand_cube.extract(
            iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))
        # ann_demand_cube = demand_cube_ext.aggregated_by('year', iris.analysis.MEDIAN)
        median_demand_cube = demand_cube_ext.collapsed('time', iris.analysis.MEDIAN)
        if period == 'present':
            temp_dict['demand', 'present', 'co2_triffid_fix'] = median_demand_cube
        else:  # when period == 'fut', calculate change in time
            temp_dict['demand', 'change', 'co2_triffid_fix'] = median_demand_cube - temp_dict[
                'demand', 'present', 'co2_triffid_fix']

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')

        for period in ['present', 'fut']:
            yr_range = [2006, 2025] if period == 'present' else [2076, 2095]
            demand_cube_monthly = monthly_demand_cube.extract(
                iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))
            supply_cube_monthly = monthly_supply_cube.extract(
                iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

            # ann_supply_cube = supply_cube_monthly.aggregated_by('year', iris.analysis.MEDIAN)
            median_supply_cube = supply_cube_monthly.collapsed('time', iris.analysis.MEDIAN)

            monthly_wsi_cube = demand_cube_monthly / supply_cube_monthly  # WSI for each season averaged over period
            # monthly_wsi_cube = common.cap_wsi_cube_abv_20(monthly_wsi_cube, threshold=20)
            median_wsi_cube = monthly_wsi_cube.collapsed('time', iris.analysis.MEDIAN)

            if period == 'present':  # for top plot
                temp_dict['supply', 'present', experiment] = median_supply_cube
                temp_dict['WSI', 'present', experiment] = median_wsi_cube

            else:  # when period == 'fut', calculate change in time
                temp_dict['supply', 'change', experiment] = median_supply_cube - temp_dict[
                    'supply', 'present', experiment]
                temp_dict['WSI', 'change', experiment] = median_wsi_cube - temp_dict['WSI', 'present', experiment]

    return temp_dict


def plot_fig5_demand_supply_wsi_present_map(plot_dir, fontsize=12):
    '''
    Plot map of present water demand, supply and WSI for simulation S4
    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plot as .png

    '''
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib as mpl
    import iris.plot as iplt

    plot_dict = preprocess_fig5_fig6_demand_supply_wsi_present_and_change_map()
    exp_list = ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']
    temp_dict = {}

    levels_dict = {'demand': np.linspace(0, 5, 11),
                   'supply': np.linspace(0, 50, 11),
                   'WSI': np.linspace(0, 1, 11)}

    cmap_dict = {'demand': 'YlOrBr',
                 'supply': 'YlGnBu'}  # WSI has discrete cbar - defined later

    title_dict = {'demand': 'Water demand',
                  'supply': 'Water supply',
                  'WSI': 'WSI',
                  }

    unit_dict = {'demand': 'm$^3$/day (x1$0^5$)',
                 'supply': 'm$^3$/day (x1$0^5$)'
                 }

    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.2, left=0.075, right=0.925)
    letters = 'abc'

    proj = ccrs.PlateCarree()
    panel = 0

    for var in ['demand', 'supply', 'WSI']:
        cube = plot_dict[var, 'present', 'co2_triffid_fix']

        panel = panel + 1
        ax = fig.add_subplot(1, 3, panel)

        if var == 'WSI':
            # make discrete cmap for WSI
            cmap = mpl.colors.ListedColormap(["white", "yellow", "orange", "chocolate", "brown", "indigo"])
            bounds = [0, 0.1, 0.2, 0.4, 1, 6]  # np.linspace(0, 1.2, 13)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cf = iplt.contourf(cube, levels=levels_dict[var], extend="max",
                               cmap=cmap, norm=norm, proj=proj)
        else:
            if var in ['demand', 'supply']:
                cube_to_plot = cube.copy()
                cube_to_plot.data = (cube_to_plot.data) / 100000
            cf = iplt.contourf(cube_to_plot, extend='max', levels=levels_dict[var],
                               cmap=plt.get_cmap(cmap_dict[var]), proj=proj)
        plt.gca().coastlines()

        cbar = plt.colorbar(cf, fraction=0.1, pad=0.05, orientation='horizontal', aspect=30)
        cbar.ax.tick_params(labelsize=fontsize)

        if panel in [1, 2]:
            plt.title('{} in {}'.format(title_dict[var], unit_dict[var]), fontsize=fontsize+1)
        elif panel == 3:
            plt.title('{}'.format(title_dict[var]), fontsize=fontsize+1)

        plt.text(0.02, 0.2, letters[panel - 1], transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                 va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))

    fname = 'fig5_demand_supply_wsi_present_map.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return
def plot_fig6_demand_supply_wsi_change_contr_factors_map(plot_dir, fontsize=12):
    '''
    Plot map of change in demand, supply and WSI between future and present periods for simulations S1 & S4,
    plus contr factors for supply and WSI

    Args:
        fontsize: int, fontsize for labels

    Returns: Saves plot as .png
    '''

    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib as mpl
    import iris.plot as iplt

    plot_dict = preprocess_fig5_fig6_demand_supply_wsi_present_and_change_map()
    exp_list = ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']
    contr_factor_list = ['CLIM', 'ALL', 'STOM', 'PLANT_PHYS', 'VEG_DIST', 'PLANT_PHYS_VEG']
    contr_factor_by_var = {}
    temp_dict = {}

    for var in ['supply', 'WSI']:
        for exp in exp_list:
            temp_dict[var, exp] = plot_dict[
                var, 'change', exp]  # to get in right format for calculating contributing factors
        contr_factor_dict = common.calc_contr_factors_map(temp_dict, var_list=[var], rel_diff=False)
        contr_factor_by_var[var] = contr_factor_dict

    unit_dict = {'demand': 'm$^3$/day (x1$0^5$)',
                 'supply': 'm$^3$/day (x1$0^5$)'
                 }
    levels_dict = {'demand': np.linspace(-1, 1, 21),
                   'supply': np.linspace(-2, 2, 21),
                   'WSI': np.linspace(-0.5, 0.5, 21)
                   }
    cmap_dict = {'demand': 'PiYG_r',
                 'supply': 'PiYG',
                 'WSI': 'PiYG_r'
                 }
    title_dict = {'demand': 'Demand change',
                  'supply': 'Supply change',
                  'WSI': 'WSI change',
                  }

    cbar_axes_dict = {1: [0.21, 0.8, 0.22, 0.015],
                      17: [0.45, 0.13, 0.22, 0.015],
                      18: [0.7, 0.13, 0.22, 0.015]}

    fig = plt.figure(figsize=(13, 9))
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.15, left=0.2, right=0.925)
    letters = 'abcdefghijklmnopqrstuvwxyz'

    proj = ccrs.PlateCarree()
    panel = 0
    letter_count = 1

    for row, factor in enumerate(contr_factor_list):
        for col, var in enumerate(['demand', 'supply', 'WSI']):
            panel = panel + 1
            ax = fig.add_subplot(6, 3, panel)

            if var == 'demand' and factor == 'CLIM':  # not needed for demand as same for all simulations
                cube_to_plot = plot_dict[var, 'change', 'co2_triffid_fix']
                cube_to_plot.data = (cube_to_plot.data) / 100000
                cf = iplt.contourf(cube_to_plot, extend="both", levels=levels_dict[var],
                                   cmap=plt.get_cmap(cmap_dict[var]), proj=proj)
                plt.gca().coastlines()
                plt.title('{} in {}'.format(title_dict[var], unit_dict[var]), fontsize=fontsize)

                cbar_ax = fig.add_axes(cbar_axes_dict[panel])  # Add axes for colorbar
                cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
                cbar.set_ticks(np.arange(-1, 1.5, 0.5))
                cbar.ax.tick_params(labelsize=fontsize-1)

                plt.text(0.04, 0.08, letters[panel - 1], transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                         va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))

                plt.text(-0.29, 0.45, 'S1. CLIM:\nSTOM', fontsize=fontsize, transform=ax.transAxes,
                         rotation='horizontal', bbox=dict(facecolor='lightgrey', alpha=0.3))

            elif var == 'demand':
                plt.axis('off')
            else:
                temp_dict_2 = contr_factor_by_var[var]
                cube_to_plot = temp_dict_2[var, factor]

                if var in ['supply']:
                    cube_to_plot.data = (cube_to_plot.data) / 100000

                cf = iplt.contourf(cube_to_plot, extend="both", levels=levels_dict[var],
                                   cmap=plt.get_cmap(cmap_dict[var]), proj=proj)
                plt.gca().coastlines()

                if panel==2:
                    plt.title('{} in {}'.format(title_dict[var], unit_dict[var]), fontsize=fontsize)
                elif panel==3:
                    plt.title('{}'.format(title_dict[var]), fontsize=fontsize)

                if panel in [17, 18]:
                    cbar_ax = fig.add_axes(cbar_axes_dict[panel])  # Add axes for colorbar
                    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
                    if var=='supply':
                        cbar.set_ticks(np.arange(-2, 3, 1))
                    else:
                        cbar.set_ticks(np.arange(-0.4, 0.6, 0.2))
                    cbar.ax.tick_params(labelsize=fontsize-1)

                plt.text(0.05, 0.1, letters[letter_count], transform=ax.transAxes, fontsize=12, fontweight='bold',
                         va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.3))
                letter_count += 1

                if col == 1 and row >= 1:  # Add labels to the left-hand column
                    contr_factor_name = common.get_contr_factor_label_maps(factor)
                    facecolor_dict = {'CLIM': 'lightgrey', 'ALL': 'lightgrey', 'STOM': 'yellowgreen',
                                      'PLANT_PHYS': 'yellowgreen', 'VEG_DIST': 'yellowgreen',
                                      'PLANT_PHYS_VEG': 'yellowgreen'}
                    if row in [1,5]: # move S4 label down to avoid demand cbar (and plant phys as big label)
                        plt.text(-0.4, 0.1, contr_factor_name, fontsize=fontsize, transform=ax.transAxes,
                                 rotation='horizontal', bbox=dict(facecolor=facecolor_dict[factor], alpha=0.3))
                    else:
                        plt.text(-0.4, 0.35, contr_factor_name, fontsize=fontsize, transform=ax.transAxes,
                                 rotation='horizontal', bbox=dict(facecolor=facecolor_dict[factor], alpha=0.3))

    fname = 'fig6_demand_supply_wsi_change_contr_factors_map'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return

def preprocess_fig7_median_wsi_by_ar6_region_to_csv(csv_dir, period='fut'):
    '''
    Calculate median WSI for each AR6 region for each experiment and save to csv
    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)

    Returns: Saves to .csv file
    '''

    import iris
    from iris import coord_categorisation
    import regionmask
    import common_functions_paper as common
    import pandas as pd

    if period == 'hist':
        yr_range = [2006, 2025]
    elif period == 'fut':
        yr_range = [2076, 2095]
    print('Using year range: {}'.format(yr_range))

    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly-neg_demand_masked.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    monthly_demand_cube = monthly_demand_cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

    ar6_land_regions = regionmask.defined_regions.ar6.land

    # # To test with one region
    # region_number = ar6_land_regions.abbrevs.index('EAS')
    # filtered_region = ar6_land_regions[region_number]

    wsi_median_dict = {}

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Calculating wsi for {}'.format(experiment))
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly-neg_demand_masked.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))
        monthly_wsi_cube = monthly_demand_cube / monthly_supply_cube

        # Also calculate global median
        global_median = monthly_wsi_cube.collapsed(['latitude', 'longitude', 'time'], iris.analysis.MEDIAN)
        wsi_median_dict[experiment, 'global'] = global_median.data

        for region in ar6_land_regions:
            cube_region = common.mask_cube_by_ar6_region(monthly_wsi_cube, region=region)
            cube_region_median = cube_region.collapsed(['latitude', 'longitude', 'time'], iris.analysis.MEDIAN)
            wsi_median_dict[experiment, region.name] = cube_region_median.data

    df = pd.DataFrame(list(wsi_median_dict.items()), columns=['keys', 'WSI Median'])

    # Split the 'keys' column into separate columns
    df[['Experiment', 'Region']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    df.drop(columns=['keys'], inplace=True)
    df = df.pivot(index='Region', columns='Experiment', values='WSI Median')

    # Save the DataFrame to a CSV
    fname = 'fig7_median_wsi_by_ar6_{}.csv'.format(period)
    df.to_csv('{}{}'.format(csv_dir, fname), index=True)
    print('Dataframe saved to {}{}'.format(csv_dir, fname))

    return df


def plot_fig7_wsi_factors_by_ar6_hbarplots(csv_dir, plot_dir, period='fut', fontsize=13):
    '''
    Plot horizontal bar plots of median WSI for each AR6 region for simulations S1 & S4 (left plot)
    and contributing factors (right plot)

    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)
        fontsize: int - size of font for labels and ticks

    Returns: Saves plot to .png file

    '''

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # .csv produced in preprocess_fig8_median_wsi_by_ar6_region_to_csv()
    df = pd.read_csv('/home/h06/jstacey/MSc/csv_files/fig7_median_wsi_by_ar6_{}.csv'.format(period))
    df = df.set_index('Region')
    df.replace('--', np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)
    df = df.astype(float)
    df = common.calc_factor_inf_df(df, percent_diff=True)
    df = df.rename(columns={'co2_triffid_fix': 'S1. CLIM: STOM', 'all_noLUC': 'S4. CLIM+CO2: STOM+VEG'})

    df = df.sort_values(by='S1. CLIM: STOM', ascending=True)
    if filter:
        print('Filtering out regions with largest WSI and adding on the global value')
        global_row = df[df.index == 'global']
        df_filter = df.tail(25)
        df = pd.concat([global_row, df_filter])
        df = df.rename(index={'global': 'Global'})

    df_factors = df[['CO2: STOM & CLIM+CO2: VEG', 'CO2: STOM+VEG', 'CO2: STOM', 'CLIM: VEG']]
    df_wsi_all = df[['S4. CLIM+CO2: STOM+VEG']]
    df_wsi_clim = df[['S1. CLIM: STOM']]

    df_wsi = pd.merge(left=df_wsi_all, right=df_wsi_clim, left_index=True,
                      right_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7.5))

    # ax1 - mean WSI by region
    df_wsi.plot(kind='barh', ax=ax1, position=0, width=0.6, color=['cornflowerblue', 'black'], edgecolor='black',
                linewidth=0.3)
    ax1.yaxis.tick_left()  # Place y-axis ticks on the left
    ax1.set_yticks(range(len(df_wsi_clim.index)))  # Set y-axis ticks
    ax1.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)  # Set y-axis labels
    ax1.set_xlim(0, 1.9)
    ax1.set_ylabel('')

    # Add grid and v lines for WSI thresholds
    ax1.axvline(x=0.2, color='black', linestyle='dotted', linewidth=2)
    ax1.text(0.2, 1, '0.2', rotation=90, va='bottom', ha='right', fontsize=fontsize)
    ax1.axvline(x=0.4, color='black', linestyle='dotted', linewidth=2)
    ax1.text(0.4, 1, '0.4', rotation=90, va='bottom', ha='right', fontsize=fontsize)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')

    # Add labels to bars for the regions where values go off the scale
    for region in ['Arabian-Peninsula']:  # , 'S.Asia']:
        s1_value = round(df_wsi.loc[region, 'S1. CLIM: STOM'])
        s4_value = round(df_wsi.loc[region, 'S4. CLIM+CO2: STOM+VEG'])
        ax1.text(1.91, df_wsi.index.get_loc(region) - 0.15, str(s1_value), va='center', ha='left')
        ax1.text(1.91, df_wsi.index.get_loc(region) + 0.4, str(s4_value), va='center', ha='left')

    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.set_xlabel('Median WSI', fontsize=fontsize)
    # reverse labels in legend so in same order as plot
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.4, -0.1), ncol=1)

    # ax2 - contributing factors
    df_factors.plot(kind='barh', ax=ax2, position=0, width=0.7, color=['seagreen', 'darkorange', 'c', 'darkslategrey'],
                    edgecolor='black', linewidth=0.3)  # 'orangered', 'maroon', 'red'])  # , color='green')
    ax2.yaxis.tick_right()  # Place y-axis ticks on the right
    ax2.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)
    ax2.set_ylabel('')
    ax2.set_xlim(-55, 20)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_xlabel('% difference', fontsize=fontsize)

    # ax2 legend with reversed order
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1], fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.6, -0.1), ncol=1)

    if period == 'fut':  # Add labels to bars for the regions where values go off the scale
        region = 'Arabian-Peninsula'
        veg_value = round(df_factors.loc[region, 'CLIM: VEG'])
        all_value = round(df_factors.loc[region, 'CO2: STOM & CLIM+CO2: VEG'])
        ax2.text(-62, df_factors.index.get_loc(region) - 0.1, str(veg_value), va='center', ha='left')
        ax2.text(-62, df_factors.index.get_loc(region) + 0.5, str(all_value), va='center', ha='left')

    # Make 'Global' label bold
    for label in ax1.get_yticklabels():
        if label.get_text() == 'Global':
            label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        if label.get_text() == 'Global':
            label.set_fontweight('bold')

    plt.tight_layout()

    if period == 'hist':
        fname = 'figS3_hbarplots_median_wsi_factors_by_ar6_hist.png'
    else:
        fname = 'fig7_hbarplots_median_wsi_factors_by_ar6_fut.png'
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))
    return

def preprocess_fig8_median_wsi_by_riverbasin_to_csv(csv_dir, period='fut'):
    '''
    Calculate median WSI for each river basin for each experiment and save to csv
    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)

    Returns: Saves to .csv file

    '''
    import iris
    from iris import coord_categorisation
    import common_functions_paper as common
    import pandas as pd
    import geopandas as gpd

    import cf_units

    if period == 'hist':
        yr_range = [2006, 2025]
    elif period == 'fut':
        yr_range = [2076, 2095]
    print('Using year range: {}'.format(yr_range))

    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly-neg_demand_masked.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    monthly_demand_cube = monthly_demand_cube.extract(
        iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

    basin_shpfile = '/data/users/jstacey/hydrobasins/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp'
    gdf = gpd.read_file(basin_shpfile)
    PFAF_list = gdf['PFAF_ID'].tolist()

    wsi_median_dict = {}

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Calculating wsi for {}'.format(experiment))
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly-neg_demand_masked.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(
            iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

        for PFAF_ID in PFAF_list:
            monthly_supply_cube_basin = common.apply_basin_mask_to_cube(monthly_supply_cube, PFAF_ID)
            monthly_supply_cube_basin = monthly_supply_cube_basin.collapsed(['latitude', 'longitude'], iris.analysis.SUM)
            monthly_demand_cube_basin = common.apply_basin_mask_to_cube(monthly_demand_cube, PFAF_ID)
            monthly_demand_cube_basin = monthly_demand_cube_basin.collapsed(['latitude', 'longitude'],
                                                                            iris.analysis.SUM)
            # Calculate median monthly WSI by basin
            monthly_wsi_cube_basin = monthly_demand_cube_basin / monthly_supply_cube_basin
            time_coord = monthly_wsi_cube_basin.coord('time')
            time_units = time_coord.units

            for cube_slice in monthly_wsi_cube_basin.slices_over('time'):
                time_point = cube_slice.coord('time').points[0]
                dt = cf_units.num2date(time_point, time_units.name, time_units.calendar)
                year = dt.year
                month = dt.month
                wsi_median_dict[experiment, PFAF_ID, year, month] = cube_slice.data

    df = pd.DataFrame(list(wsi_median_dict.items()), columns=['keys', 'Median'])

    # Split the 'keys' column into separate columns
    df[['Experiment', 'PFAF_ID', 'year', 'month']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    df.drop(columns=['keys'], inplace=True)
    df = df.pivot_table(index=['PFAF_ID', 'year', 'month'], columns='Experiment', values='Median')

    # Save the DataFrame to a CSV
    fname = 'fig8_median_monthly_wsi_{}_by_basins.csv'.format(period)
    df.to_csv('{}{}'.format(csv_dir, fname), index=True)
    print('Dataframe saved to {}{}'.format(csv_dir, fname))

    return df


def preprocess_fig8_median_wsi_by_basin(plot_list, csv_dir):
    '''
    Preprocess data for plotting fig9 - median WSI by basin
    Args:
        experiment_list: List of experimetns to plot

    Returns: DataFrame with median WSI by basin for each experiment

    '''
    import pandas as pd
    import geopandas as gpd
    import numpy as np

    import common_functions_paper as common

    period = 'fut'
    exp_list = ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']

    print('Loading basin shapes and csvs')
    # load basins from original file to keep the crs
    basin_shpfile = '/data/users/jstacey/hydrobasins/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp'
    basins_df = gpd.read_file(basin_shpfile)
    basins_df = basins_df[['PFAF_ID', 'geometry']]  # .head(3) # for testing
    basins_df['PFAF_ID'] = basins_df['PFAF_ID'].astype(int)

    df = pd.read_csv('{}/fig8_median_monthly_wsi_fut_by_basins.csv'.format(csv_dir))
    df['PFAF_ID'] = df['PFAF_ID'].astype(int)

    df = df.replace('--', 0)

    for exp in exp_list:
        df[exp] = df[exp].astype(float)

    df_by_region = df.groupby(['PFAF_ID'], sort=True)[exp_list].median()

    df_by_region = basins_df.merge(df_by_region, on='PFAF_ID', how='right')
    df_by_region = common.calc_factor_inf_df(df_by_region, percent_diff=True)
    df_by_region = df_by_region.replace(np.nan, 0)

    for col in plot_list:
        color_dict = common.get_basin_colour_dict(col, reldiff=True, seasonal_flag=False)
        new_var_name = 'color_{}'.format(col)
        df_by_region = df_by_region.assign(**{new_var_name: common.make_color_list_for_wsi(df_by_region, col, color_dict)})

    return df_by_region

def plot_fig8_wsi_maps_by_basin_reldiff(period, csv_dir, plot_dir, fontsize=12):
    '''
    Plot maps of median WSI by basin for each experiment
    Args:
        period: 'hist' for 2006-2025 or 'fut' for 2076-2095
        fontsize: int, fontsize for labels

    Returns: Saves plot to .png file

    '''

    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    plot_list = ['co2_triffid_fix', 'all_noLUC', 'CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']

    df = preprocess_fig8_median_wsi_by_basin(plot_list, csv_dir=csv_dir)

    print('Plotting time')
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=0.01, wspace=0.05, top=0.95, bottom=0.01, left=0.075, right=0.8)
    letters = 'abcdef'

    panel = 0
    for i, plot in enumerate(plot_list):
        print('Now plotting {}'.format(plot))

        handle_list = []
        panel += 1
        ax = fig.add_subplot(3, 2, panel)
        color_dict = common.get_basin_colour_dict(plot, reldiff=True, seasonal_flag=False)

        for j, (range_min, range_max) in enumerate(color_dict):
            color = color_dict[(range_min, range_max)]
            if j == 0 and panel == 2: # to make label 5+ for top WSI values
                label = '{}+'.format(range_min)
            else:
                label = '{} to {}'.format(range_min, range_max)
            df_temp = df.loc[(df['color_{}'.format(plot)] == color)]

            if df_temp.empty:
                print('No entries for color {} between {} and {}'.format(color, range_min, range_max))
            else:
                df_temp.plot(
                    ax=ax,
                    color=color,
                    edgecolor='black',
                    label=label,
                    linewidth=0.7
                )
                plt.gca().set_aspect('equal')
            patch_for_legend = mpatches.Patch(color=color, label=label)
            handle_list.append(patch_for_legend)

        color_dict.clear()  # for next plot
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)  # remove lat/lon labels
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)

        if panel == 2:
            ax.legend(handles=handle_list, bbox_to_anchor=(1.16, 1.1), ncol=1, loc='upper center', fontsize=fontsize,
                      title='Median WSI', title_fontsize=fontsize)
        if panel == 4:
            ax.legend(handles=handle_list, bbox_to_anchor=(1.16, 0.6), ncol=1, loc='upper center', fontsize=fontsize,
                      title='% difference', title_fontsize=fontsize)

        if plot in ['co2_triffid_fix', 'all_noLUC']:
            plot = common.get_experiment_label(plot)
        ax.set_title(plot, fontsize=fontsize)  # get_title_for_wsi_basin_plots(col_name, year_range), fontsize=10)
        plt.text(0.01, 0.97, letters[panel - 1], transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                 va='top', ha='left', bbox=dict(facecolor='white', alpha=1))

    fname = 'fig8_median_wsi_by_basins_{}_reldiff.png'.format(period)
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))

    return


def table2_output_basins_pop_incr_decr_to_csv(plot_dir, remove_non_water_scarce_basins=False):
    '''
    Calculate number of basins and population that increase and decrease WSI by a given threshold for each factor
    Args:
        remove_non_water_scarce_basins: boolean, if True, remove basins that are not water scarce in both dynamic and fixed experiments

    Returns: Saves table to .csv file

    '''
    import pandas as pd
    period = 'fut'
    factor_list = ['CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']
    df = preprocess_fig8_median_wsi_by_basin(factor_list)
    print(df.columns)

    if remove_non_water_scarce_basins:
        print('IF USING - NEED TO CHANGE THIS TO BE IN LINE WITH FIG 9 PREPROCESSING AND INC. OTHER SIMULATIONS')
        print('Removing non water scarce basins where water scarce in both dynamic and fixed experiments')
        color_exp_list = ['color_all_noLUC', 'color_co2_triffid_fix']
        original_length = len(df)
        df = df[~df[color_exp_list].isin(['white']).all(axis=1)]
        new_length = len(df)
        rows_removed = original_length - new_length
        print(f"Number of rows removed: {rows_removed}")

    table_dict = {'Factor': [], 'Threshold': [], 'basin_ct_incr': [], 'basin_ct_decr': [], 'pop_ct_incr': [], 'pop_ct_decr': [],
                  'pop_perc_incr': [], 'pop_perc_decr': []}

    for threshold in [5, 10, 20, 30, 40, 50]:
        for factor in factor_list:
            table_dict = common.calc_basins_incr_decr(df, factor, threshold, table_dict)

    df = pd.DataFrame(table_dict)
    fname = 'table2_basin_pop_count.csv'
    df.to_csv('{}{}'.format(plot_dir, fname))
    print('{}{}'.format(plot_dir, fname))

    return


def preprocessing_fig9_median_wsi_by_basin_anncycle(csv_dir):
    '''
    Preprocess data for plotting fig10 - median WSI by basin for each month
    Returns: DataFrame with median WSI by basin for each experiment

    '''
    import pandas as pd
    import geopandas as gpd
    import numpy as np

    import common_functions_paper as common

    period = 'fut'
    exp_list = ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']

    print('Loading basin monthly WSI csv')
    df = pd.read_csv('{}/fig8_median_monthly_wsi_fut_by_basins.csv'.format(csv_dir))
    df['PFAF_ID'] = df['PFAF_ID'].astype(int)

    df = df.replace('--', 0)

    for exp in exp_list:
        df[exp] = df[exp].astype(float)

    df_by_region_anncycle = df.groupby(['PFAF_ID', 'month'], sort=True)[exp_list].median()
    df_by_basin_anncycle = common.calc_factor_inf_df(df_by_region_anncycle, percent_diff=True)
    df_by_basin_anncycle = df_by_basin_anncycle.replace(np.nan, 0)

    return df_by_basin_anncycle

def plot_fig9_wsi_anncycle_singlebasin(basin_id, basin_name, csv_dir, plot_dir, font_size=20):
    '''
    Plot WSI annual cycle for selected basins - Fig. 10 then put together in powerpoint
    Args:
        basin_id: int, PFAF_ID of basin
        basin_name: str, name of basin for filename
        font_size: int, size of font for labels and ticks

    Returns: Saves plot to .png file

    '''
    import matplotlib.pyplot as plt
    import common_functions_paper as common

    df = preprocessing_fig9_median_wsi_by_basin_anncycle(csv_dir)
    df = df.loc[basin_id]

    fig = plt.figure(figsize=(3, 3))
    fig.subplots_adjust(hspace=0.05, bottom=0.05, top=0.95, left=0.015, right=0.99)
    ax1 = fig.add_subplot(2, 1, 1)
    for exp in ['co2_triffid_fix', 'co2_fix_noLUC', 'triffid_fix', 'all_noLUC']:
        color = common.get_experiment_color_dict(exp)
        plt.plot(df.index, df[exp], label=exp, color=color)
    ax1.set_ylabel('WSI', fontsize=font_size)
    ax1.axhline(y=0.2, color='dimgray', linestyle='--')
    ax1.axhline(y=0.4, color='dimgray', linestyle='--')

    pop_count = common.get_fut_pop_count_by_basin(basin_id, in_millions=True)

    ax1.set_title('Pop. {:,} million'.format(pop_count), fontsize=font_size+2)
    plt.tick_params(axis='x', which='both', labelbottom=False)  # don't plot x-axis on top plot
    plt.yticks(fontsize=font_size)

    # Bottom panel: contr_factor subplot
    ax2 = fig.add_subplot(2, 1, 2)
    for contr_factor in ['CLIM: VEG', 'CO2: STOM', 'CO2: STOM+VEG', 'CO2: STOM & CLIM+CO2: VEG']:
        color = common.get_contr_factor_color_dict(contr_factor)
        plt.plot(df.index, df[contr_factor], label=contr_factor, color=color)
    ax2.set_ylabel('% diff.', fontsize=font_size)
    ax2.axhline(y=0, color='dimgray', linestyle='--')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    fname = 'fig9_WSI_anncycle_fut_{}.png'.format(basin_name)
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))
    return

def plot_fig9_basin_shapes(basin_dict, plot_dir):
    '''
    Plot global map highlighting selected basins
    Args:
        basin_dict: dictionary, {PFAF_ID: basin_name}

    Returns: Saves plot to .png file

    '''


    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import geopandas as gpd

    shpfile = '/data/users/jstacey/hydrobasins/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp'
    df = gpd.read_file(shpfile)

    fig = plt.figure(figsize=(25, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cf.BORDERS, linestyle=':', alpha=1)
    ax.add_feature(cf.LAND, zorder=0)
    ax.add_feature(cf.OCEAN, zorder=0, alpha=.4)
    ax.coastlines("50m", linewidth=1.0)

    for PFAF_ID in basin_dict:
        basin = df[df['PFAF_ID'] == PFAF_ID]
        basin.plot(ax=ax, edgecolor='black', linewidth=1, alpha=.8) # also assign colour
        #if inc_PFAF_label:
        #    basin.apply(lambda x: ax.annotate(text=x['PFAF_ID'], xy=x.geometry.centroid.coords[0], ha='center',
        #                                 color='black', path_effects=[pe.withStroke(linewidth=2, foreground="white")]), axis=1)

    ax.set_extent((-150, 160, -60, 80), crs=ccrs.PlateCarree())

    fname = "fig9_basins_map.png"
    plt.savefig('{}{}'.format(plot_dir, fname), dpi=300,
            bbox_inches='tight', facecolor='white')
    print('Plot saved to: {}{}'.format(plot_dir, fname))
    return

def main():
    '''
    Run all functions to produce figures and tables for paper
    '''
    PLOT_DIR = '/home/h06/jstacey/MSc/plots/paper/' # directory to output plots
    CSV_OUTPUT_DIR = '/home/h06/jstacey/MSc/csv_files/' # directory to output .csvs

    # # # FIGURE 1
    # plot_fig1a_global_input_vars_timeseries(plot_dir=PLOT_DIR)
    # plot_fig1b_global_change_temp_ppn_map(fontsize=14, plot_dir=PLOT_DIR)
    #
    # # # # FIGURE 2
    # plot_fig2_global_multivar_timeseries(var_list_no=1, fontsize=18, plot_dir=PLOT_DIR)
    # plot_fig2_global_multivar_timeseries(var_list_no=2, fontsize=18, plot_dir=PLOT_DIR)
    #
    # # # # FIGURE 3
    # plot_fig3_global_change_multivar_factors_map(plot_dir=PLOT_DIR)
    # #
    # # # FIGURE 4
    # plot_fig4_global_demand_supply_wsi_timeseries(fontsize=16, plot_dir=PLOT_DIR) # Run on SPICE
    # plot_fig4_as_key_fig(fontsize=14, plot_dir=PLOT_DIR)
    # #
    # # # FIGURE 5
    # plot_fig5_demand_supply_wsi_present_map(fontsize=14, plot_dir=PLOT_DIR)
    #
    # # FIGURE 6
    # plot_fig6_demand_supply_wsi_change_contr_factors_map(fontsize=14, plot_dir=PLOT_DIR)

    ## FIGURE 7 (and S4)
    #preprocess_fig7_median_wsi_by_ar6_region_to_csv(period='fut', csv_dir=CSV_OUTPUT_DIR)
    #plot_fig7_wsi_factors_by_ar6_hbarplots(period='fut', fontsize=13, plot_dir=PLOT_DIR, csv_dir=CSV_OUTPUT_DIR)

    ## FIGURE S4 - run here as same function as Fig 7 but hist period
    #preprocess_fig7_median_wsi_by_ar6_region_to_csv(period='hist', csv_dir=CSV_OUTPUT_DIR)
    #plot_fig7_wsi_factors_by_ar6_hbarplots(period='hist', fontsize=13, plot_dir=PLOT_DIR)

    ## FIGURE 8
    preprocess_fig8_median_wsi_by_riverbasin_to_csv(period='fut', csv_dir=CSV_OUTPUT_DIR)
    plot_fig8_wsi_maps_by_basin_reldiff(period='fut', fontsize=14, csv_dir=CSV_OUTPUT_DIR, plot_dir=PLOT_DIR)

    ## TABLE 2
    table2_output_basins_pop_incr_decr_to_csv(remove_non_water_scarce_basins=False, plot_dir=PLOT_DIR)

    ## FIGURE 9
    select_basins_dict = common.get_select_basins()
    for basin_id in select_basins_dict: #{294: 'Tigris-Euphrates'}:
       plot_fig9_wsi_anncycle_singlebasin(basin_id=basin_id, basin_name=select_basins_dict[basin_id], font_size=17,
                                          csv_dir=CSV_OUTPUT_DIR, plot_dir=PLOT_DIR)
    plot_fig9_basin_shapes(select_basins_dict, plot_dir=PLOT_DIR) # map in the middle of Figure

    return

if __name__ == '__main__':
    main()
