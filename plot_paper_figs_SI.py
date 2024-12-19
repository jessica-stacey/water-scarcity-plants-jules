import iris
import iris.plot as iplt
import numpy as np
import matplotlib.pyplot as plt
import common_functions_paper as common

plt.rcParams['font.family'] = 'Arial'


def plot_figS1_pop_data_map(yr_range, fontsize=12):
    import matplotlib.pyplot as plt
    import numpy as np
    import iris.plot as iplt

    import iris
    import common_functions as common
    import cf_units

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
    cube = common.extract_years(cube, yr_range)
    cube = cube.collapsed('time', iris.analysis.MEAN)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    # Mask sea points and non populated areas
    masked_population_data = np.ma.masked_where(cube.data == 0, cube.data)

    # Replace the data in the population cube with the masked data
    cube.data = masked_population_data

    fig = plt.figure(figsize=(6.5, 3))

    levels = np.linspace(0, 500000, 21)
    cmap = plt.get_cmap('RdPu')
    cf = iplt.contourf(cube, cmap=cmap, extend='max', levels=levels)
    cbar = plt.colorbar(cf, orientation='horizontal') # fraction=0.06, pad=0.01,
    cbar.ax.tick_params(labelsize=fontsize-1)#, rotation=90)
    plt.title('{} - {}'.format(yr_range[0], yr_range[1]), fontsize=fontsize)
    plt.gca().coastlines()

    fname = 'figS1_pop_map_{}-{}.png'.format(yr_range[0], yr_range[1])
    plt.savefig('/home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname), dpi=300,
                bbox_inches='tight', facecolor='white')
    print('Saving pop map to /home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname))
    return

def preprocess_figS4a_calc_month_count_wsi_by_ar6_region_to_csv(period='fut', threshold = 0.4):
    '''
    Figure S5a (spin off from Fig 7)
    Preprocess data
    Uses regional demand / regional supply

    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)
        threshold: int - WSI threshold for severe water scarcity

    Returns: Saves dataframe as .csv

    '''

    import iris
    from iris import coord_categorisation
    import regionmask
    import common_functions_paper as common
    import pandas as pd
    import numpy as np

    if period == 'hist':
        yr_range = [2006, 2025]
    elif period == 'fut':
        yr_range = [2076, 2095]
    print('Using year range: {}'.format(yr_range))

    ar6_land_regions = regionmask.defined_regions.ar6.land

    # # To test with one region
    # region_number = ar6_land_regions.abbrevs.index('EAS')
    # ar6_land_regions = [ar6_land_regions[region_number]]

    # Get demand data
    demand_by_region_dict = {}
    demand_fname = '/data/users/jstacey/water_demand/ISIMIP2/demand_HADGEM2-ES_H08_ssp2_2006-2099_monthly-neg_demand_masked.nc'
    monthly_demand_cube = iris.load_cube(demand_fname)
    iris.coord_categorisation.add_year(monthly_demand_cube, 'time')
    monthly_demand_cube = monthly_demand_cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

    # Get supply data
    for region in ar6_land_regions:
        monthly_demand_cube_region = common.mask_cube_by_ar6_region(monthly_demand_cube, region=region)
        demand_by_region_dict[region.name] = monthly_demand_cube_region.collapsed(['latitude', 'longitude'],
                                                                                  iris.analysis.SUM)

    ws_month_count_by_region_dict = {}

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Calcualting wsi for {}'.format(experiment))
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly-neg_demand_masked.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))

        for region in ar6_land_regions:
            monthly_supply_cube_region = common.mask_cube_by_ar6_region(monthly_supply_cube, region=region)
            monthly_supply_cube_region = monthly_supply_cube_region.collapsed(['latitude', 'longitude'],
                                                                              iris.analysis.SUM)

            # Calcuate WSI by region and number of water scarce months
            monthly_wsi_by_region = demand_by_region_dict[region.name] / monthly_supply_cube_region

            ws_month_count_by_region = np.sum(monthly_wsi_by_region.data >= threshold)
            ws_month_count_by_region_dict[experiment, region.name] = ws_month_count_by_region

    # Convert to df
    df = pd.DataFrame(list(ws_month_count_by_region_dict.items()), columns=['keys', 'month_count'])

    # Split the 'keys' column into separate columns
    df[['Experiment', 'Region']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    df.drop(columns=['keys'], inplace=True)
    df = df.pivot(index='Region', columns='Experiment', values='month_count')

    # Save the DataFrame to a CSV
    fname = 'figS4a_region_mean_wsi_month_count_{}_{}p{}.csv'.format(period,str(threshold)[0], str(threshold)[2])
    df.to_csv('/home/h06/jstacey/MSc/csv_files/{}'.format(fname),
              index=True)
    print(
        'Dataframe saved to /home/h06/jstacey/MSc/csv_files/{}'.format(fname))

    return df

def plot_figS4a_wsi_month_count_factors_by_ar6_hbarplots(period='fut', filter=False, fontsize=13, threshold=0.4):
    '''
    Figure S5a (spin off from Fig 7)
    Preprocess horizontal bar plots of number of months in severe water scarcity (WSI > 0.4) for each AR6 region for
    simulations S1 & S3 (left plot) and contributing factors (right plot)

    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)
        threshold: int - WSI threshold for severe water scarcity

    Returns: Saves plot to .png file

    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import common_functions_paper as common

    df = pd.read_csv('/home/h06/jstacey/MSc/csv_files/figS4a_region_mean_wsi_month_count_{}_{}p{}.csv'.format(period, str(threshold)[0], str(threshold)[2]))
    df = df.set_index('Region')
    df.replace('--', np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)
    df = df.astype(float)
    df = df.div(20)  # Get average water scarce month count per year

    df = common.calc_factor_inf_df(df, percent_diff=False)
    df = df.rename(columns={'co2_triffid_fix': 'S1. CLIM: STOM', 'all_noLUC': 'S4. CLIM+CO2: STOM+VEG'})

    df_sorted = df.sort_values(by='S1. CLIM: STOM', ascending=True)
    if filter:
        df_sorted = df_sorted.tail(25) # take the 25 largest

    df_factors = df_sorted[['CO2: STOM & CLIM+CO2: VEG', 'CO2: STOM+VEG', 'CO2: STOM', 'CLIM: VEG']]
    df_wsi_all = df_sorted[['S4. CLIM+CO2: STOM+VEG']]
    df_wsi_clim = df_sorted[['S1. CLIM: STOM']]

    df_wsi = pd.merge(left=df_wsi_all, right=df_wsi_clim, left_index=True,
                      right_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 6.5))

    # ax1 - mean WSI by region
    df_wsi.plot(kind='barh', ax=ax1, position=0, width=0.6, color=['cornflowerblue', 'black'], edgecolor='black',
                linewidth=0.3)
    ax1.yaxis.tick_left()  # Place y-axis ticks on the left
    ax1.set_yticks(range(len(df_wsi_clim.index)))  # Set y-axis ticks
    ax1.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)  # Set y-axis labels
    ax1.set_xlim(0, 12)
    ax1.set_ylabel('')

    # Add grid and v lines for WSI thresholds
    ax1.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.set_xlabel('No. of Months', fontsize=fontsize)

    # reverse labels in legend so in same order as plot
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.4, -0.1), ncol=1)

    # ax2 - contributing factors
    df_factors.plot(kind='barh', ax=ax2, position=0, width=0.7, color=['seagreen', 'darkorange', 'c', 'darkslategrey'],
                    edgecolor='black', linewidth=0.3)
    ax2.yaxis.tick_right()  # Place y-axis ticks on the right
    ax2.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)
    ax2.set_ylabel('')
    ax2.set_xlim(-2, 0.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_xlabel('Difference', fontsize=fontsize)

    # ax2 legend with reversed order
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1], fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.6, -0.1), ncol=1)

    plt.tight_layout()

    fname = 'figS4a_barplots_WS_month_count_factors_by_ar6_{}_{}p{}'.format(period,
                                                                             str(threshold)[0], str(threshold)[2])
    plt.savefig('/home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname))
    print('Plot saved to /home/h06/jstacey/MSc/plots/paper/{}'.format(fname))
    return


def preprocess_figS4b_calc_percent_area_wsi_by_ar6_region_to_csv(period='fut', threshold=0.4):
    '''
    Figure S5b (spin off from Fig 7)
    Preprocess data

    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)
        threshold: int - WSI threshold for severe water scarcity

    Returns: Saves dataframe as .csv

    '''

    import iris
    from iris import coord_categorisation
    import regionmask
    import common_functions_paper as common
    import pandas as pd
    import numpy as np

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
    # ar6_land_regions = [ar6_land_regions[region_number]]

    percent_area_dict = {}

    for experiment in ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']:
        print('Calcualting wsi for {}'.format(experiment))
        supply_fname = '/data/users/jstacey/water_demand/ISIMIP2/supply_HADGEM2-ES_{}_2006-2099_runoff_monthly-neg_demand_masked.nc'.format(
            experiment)
        monthly_supply_cube = iris.load_cube(supply_fname)
        iris.coord_categorisation.add_year(monthly_supply_cube, 'time')
        monthly_supply_cube = monthly_supply_cube.extract(iris.Constraint(year=lambda cell: yr_range[0] <= cell.point <= yr_range[1]))
        monthly_wsi_cube = monthly_demand_cube / monthly_supply_cube

        for region in ar6_land_regions:
            cube_region = common.mask_cube_by_ar6_region(monthly_wsi_cube, region=region)

            # Count number of gridcells
            cube_region_filtered = cube_region.data >= threshold
            num_WSI_gps = np.sum(cube_region_filtered)
            mean_monthly_gps = num_WSI_gps / (12 * 20)  # 240 total months in 20 years

            # Count the total number of grid points in the region
            total_grid_points = np.ma.count(cube_region.data)
            total_monthly_gps = total_grid_points / (12 * 20)

            # Calculate the percentage
            percent_area_by_month = np.round((mean_monthly_gps / total_monthly_gps) * 100, 2)
            percent_area_dict[experiment, region.name] = percent_area_by_month

    df = pd.DataFrame(list(percent_area_dict.items()), columns=['keys', 'percent_area'])

    # Split the 'keys' column into separate columns
    df[['Experiment', 'Region']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    df.drop(columns=['keys'], inplace=True)
    df = df.pivot(index='Region', columns='Experiment', values='percent_area')

    df = common.calc_factor_inf_df(df, percent_diff=False)

    # Save the DataFrame to a CSV
    fname = 'figS4b_mean_percent_area_gp_{}_{}p{}.csv'.format(period, str(threshold)[0], str(threshold)[2])
    df.to_csv('/home/h06/jstacey/MSc/csv_files/{}'.format(fname), index=True)
    print('Dataframe saved to /home/h06/jstacey/MSc/csv_files/{}'.format(fname))
    return df

def plot_figS4b_wsi_percent_area_by_ar6_hbarplots(period='fut', filter=False, fontsize=13, threshold=0.4):
    '''
    Figure S5a (spin off from Fig 7)
    Preprocess horizontal bar plots of number of months in severe water scarcity (WSI > 0.4) for each AR6 region for
    simulations S1 & S3 (left plot) and contributing factors (right plot)

    Args:
        period: 'hist' (for 2006 - 2025) or 'fut' (for 2076 - 2095)
        filter: bool - whether to filter out the top 25 regions with highest values
        fontsize: int - size of font for labels and ticks
        threshold: int - WSI threshold for severe water scarcity

    Returns: Saves plot to .png file

    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import common_functions_paper as common

    df = pd.read_csv('/home/h06/jstacey/MSc/csv_files/figS4b_mean_percent_area_gp_{}_{}p{}.csv'.format(period,
                                                                                                str(threshold)[0],
                                                                                                str(threshold)[2]))
    df = df.set_index('Region')
    df.replace('--', np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)
    df = df.astype(float)
    df = df.rename(columns={'co2_triffid_fix': 'S1. CLIM: STOM', 'all_noLUC': 'S4. CLIM+CO2: STOM+VEG'})

    df_sorted = df.sort_values(by='S1. CLIM: STOM', ascending=True)
    if filter:
        df_sorted = df_sorted.tail(25) # take the 25 largest

    df_factors = df_sorted[['CLIM: VEG', 'CO2: STOM', 'CO2: STOM+VEG', 'CO2: STOM & CLIM+CO2: VEG']]
    df_wsi_all = df_sorted[['S4. CLIM+CO2: STOM+VEG']]
    df_wsi_clim = df_sorted[['S1. CLIM: STOM']]

    df_wsi = pd.merge(left=df_wsi_clim, right=df_wsi_all, left_index=True,
                      right_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 6.5))

    # ax1 - mean WSI by region
    df_wsi.plot(kind='barh', ax=ax1, position=0, width=0.6, color=['cornflowerblue', 'black'], edgecolor='black',
                linewidth=0.3)
    ax1.yaxis.tick_left()  # Place y-axis ticks on the left
    ax1.set_yticks(range(len(df_wsi_clim.index)))  # Set y-axis ticks
    ax1.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)  # Set y-axis labels
    ax1.set_xlim(0, 80)
    ax1.set_ylabel('')

    # Add grid and v lines for WSI thresholds
    ax1.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')

    ax1.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.05, -0.1), ncol=1)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.set_xlabel('% area', fontsize=fontsize)

    # ax2 - contributing factors
    df_factors.plot(kind='barh', ax=ax2, position=0, width=0.7, color=['seagreen', 'darkorange', 'c', 'darkslategrey'],
                    edgecolor='black', linewidth=0.3)  # 'orangered', 'maroon', 'red'])  # , color='green')
    ax2.yaxis.tick_right()  # Place y-axis ticks on the right
    ax2.set_yticklabels(df_wsi_clim.index, fontsize=fontsize)
    # ax2.set_yticklabels('')
    ax2.set_ylabel('')
    ax2.set_xlim(-10, 2)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.6, color='grey')
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_xlabel('Difference', fontsize=fontsize)
    ax2.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.9, -0.1), ncol=1)

    plt.tight_layout()

    fname = 'figS4b_barplots_percent_area_by_ar6_{}_{}p{}.png'.format(period, str(threshold)[0], str(threshold)[2])
    plt.savefig('/home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname))
    print('Plot saved to /home/h06/jstacey/MSc/plots/paper/{}'.format(fname))
    return


def preprocess_figS5_wsi_month_count_by_basin(plot_list):
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

    df = pd.read_csv('/home/h06/jstacey/MSc/csv_files/fig9_median_monthly_wsi_fut_by_basins.csv')
    df['PFAF_ID'] = df['PFAF_ID'].astype(int)

    df = df.replace('--', 0)

    for exp in exp_list:
        df[exp] = df[exp].astype(float)
        df[exp] = df[exp].clip(upper=20)

    exp_list = ['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']

    threshold = 0.4
    df_WScount = df.groupby('PFAF_ID')[exp_list].apply(lambda x: (x > threshold).sum())
    df_WScount = df_WScount / 20

    df_by_region = basins_df.merge(df_WScount, on='PFAF_ID', how='right')
    df_by_region = common.calc_factor_inf_df(df_by_region, percent_diff=False)
    df_by_region = df_by_region.replace(np.nan, 0)

    for col in plot_list:
        color_dict = common.get_basin_colour_dict_month_count(col)
        new_var_name = 'color_{}'.format(col)
        df_by_region = df_by_region.assign(**{new_var_name: common.make_color_list_for_wsi(df_by_region, col, color_dict)})

    return df_by_region


def plot_figS5_wsi_month_count_maps_by_basin(period='fut', fontsize=12):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    plot_list = ['co2_triffid_fix', 'all_noLUC', 'CO2: STOM', 'CLIM: VEG', 'CO2: STOM+VEG', 'CO2: STOM & CLIM+CO2: VEG']

    df = preprocess_figS5_wsi_month_count_by_basin(plot_list)

    print('Plotting time')
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=0.01, wspace=0.05, top=0.95, bottom=0.01, left=0.075, right=0.8)

    panel = 0
    for i, plot in enumerate(plot_list):
        print('Now plotting {}'.format(plot))

        handle_list = []
        panel += 1
        ax = fig.add_subplot(3, 2, panel)
        color_dict = common.get_basin_colour_dict_month_count(plot)

        for (range_min, range_max) in color_dict:
            color = color_dict[(range_min, range_max)]
            label = '{} to {}'.format(range_min, range_max)
            df_temp = df.loc[(df['color_{}'.format(plot)] == color)]

            if df_temp.empty:
                print('No entries for color {} between {} and {}'.format(color, range_min, range_max))
            else:
                df_temp.plot(ax=ax, color=color, label=label, edgecolor='black', linewidth=0.7)
                plt.gca().set_aspect('equal')
            patch_for_legend = mpatches.Patch(color=color, label=label)
            handle_list.append(patch_for_legend)

        color_dict.clear()  # for next plot
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False,
                        bottom=False)  # remove lat/lon labels
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)

        if panel == 2:
            ax.legend(handles=handle_list, bbox_to_anchor=(1.15, 1.1), ncol=1, loc='upper center', fontsize=fontsize,
                      title='No. of months', title_fontsize=fontsize)
        if panel == 4:
            ax.legend(handles=handle_list, bbox_to_anchor=(1.18, 0.5), ncol=1, loc='upper center', fontsize=fontsize,
                      title='Difference', title_fontsize=fontsize)

        if plot in ['co2_triffid_fix', 'all_noLUC']:
            plot = common.get_experiment_label(plot)
        ax.set_title(plot, fontsize=fontsize+2)  # get_title_for_wsi_basin_plots(col_name, year_range), fontsize=10)

        if plot in ['CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']:
            # Add text box showing number of basins increasing and decreasing
            num_incr = (df[plot] >= 0.5).sum()
            num_decr = (df[plot] <= -0.5).sum()
            text_str = f"Inc.: {num_incr}\nDec.: {num_decr}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='bottom', bbox=props)

    fname = 'figS5_wsi_month_count_by_basins_{}.png'.format(period)
    plt.savefig('/home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname), dpi=300,
                bbox_inches='tight', facecolor='white')
    print('Saving plot to /home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname))

    return


def preprocess_figS6_median_wsi_by_basin_plus_seasons(plot_list):
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

    df = pd.read_csv(
        '/home/h06/jstacey/MSc/csv_files/fig9_median_monthly_wsi_fut_by_basins.csv')
    df['PFAF_ID'] = df['PFAF_ID'].astype(int)

    df = df.replace('--', 0)

    for exp in exp_list:
        df[exp] = df[exp].astype(float)
        df[exp] = df[exp].clip(upper=20)

    print('Mapping months to seasons')
    month_to_season = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
                       7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    df['season'] = df['month'].map(month_to_season)
    df_by_region = df.groupby(['PFAF_ID', 'season'], sort=True)[exp_list].median()

    df_by_region = df_by_region.reset_index()
    df_by_region = basins_df.merge(df_by_region, on='PFAF_ID', how='right')
    df_by_region.set_index(['PFAF_ID', 'season'], inplace=True)
    print('**** after merge', df_by_region)
    df_by_region = common.calc_factor_inf_df(df_by_region, percent_diff=True)
    df_by_region = df_by_region.replace(np.nan, 0)

    for factor in ['CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']:
        # clip huge values as they will all going into a 40+ category
        df_by_region[factor] = df_by_region[factor].clip(lower=-99, upper=99)

    for col in plot_list:
        color_dict = common.get_basin_colour_dict(col, reldiff=True)
        new_var_name = 'color_{}'.format(col)
        df_by_region = df_by_region.assign(
            **{new_var_name: common.make_color_list_for_wsi(df_by_region, col, color_dict)})

    print('Masking where all experiments have median WSI < 0.05')
    df_by_region.loc[
        (df_by_region[['all_noLUC', 'co2_fix_noLUC', 'triffid_fix', 'co2_triffid_fix']] < 0.05).all(axis=1), [
            'color_co2_triffid_fix', 'color_all_noLUC', 'color_CO2: STOM', 'color_CLIM: VEG', 'color_CO2: STOM+VEG',
            'color_CO2: STOM & CLIM+CO2: VEG']] = 'white'

    return df_by_region

def plot_figS6_wsi_maps_by_basin_reldiff_by_seasons(period='fut', fontsize=12):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    plot_list = ['co2_triffid_fix', 'all_noLUC', 'CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']
    seas_list = ['DJF', 'MAM', 'JJA', 'SON']

    df = preprocess_figS6_median_wsi_by_basin_plus_seasons(plot_list)

    print('Plotting time')
    fig = plt.figure(figsize=(16.5, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.01, top=0.95, bottom=0.01, left=0.1, right=0.8)
    letters = 'abcdefghijklmnopqrstuvwxyz'

    panel = 0
    for row, plot in enumerate(plot_list):
        print('Now plotting {}'.format(plot))
        handle_list = []
        for col, seas in enumerate(seas_list):

            panel += 1
            ax = fig.add_subplot(len(plot_list), len(seas_list), panel)
            color_dict = common.get_basin_colour_dict(plot, reldiff=True)
            handle_list = []

            for (range_min, range_max) in color_dict:
                color = color_dict[(range_min, range_max)]
                label = '{} to {}'.format(range_min, range_max)
                df_seas = df.xs(seas, level='season')
                df_temp = df_seas.loc[(df_seas['color_{}'.format(plot)] == color)]

                if df_temp.empty:
                    print('No entries for color {} between {} and {}'.format(color, range_min, range_max))
                else:
                    df_temp.plot(ax=ax, color=color, label=label, edgecolor='black', linewidth=0.7)
                    plt.gca().set_aspect('equal')
                patch_for_legend = mpatches.Patch(color=color, label=label)
                handle_list.append(patch_for_legend)

            color_dict.clear()  # for next plot
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False,
                            bottom=False)  # remove lat/lon labels
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 85)

            if panel == 4:
                ax.legend(handles=handle_list, bbox_to_anchor=(1.27, 1.1), ncol=1, loc='upper center', fontsize=fontsize,
                          title='Median WSI', title_fontsize=fontsize)
            if panel == 12:
                ax.legend(handles=handle_list, bbox_to_anchor=(1.27, 0.3), ncol=1, loc='upper center', fontsize=fontsize,
                          title='% difference', title_fontsize=fontsize)

            if row==0: # Set season title on top row only
                ax.set_title('{}'.format(seas), fontsize=fontsize+2)

            #plt.text(0.01, 0.97, letters[panel - 1], transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
            #         va='top', ha='left', bbox=dict(facecolor='white', alpha=1))

            if plot in ['CO2: STOM', 'CO2: STOM+VEG', 'CLIM: VEG', 'CO2: STOM & CLIM+CO2: VEG']:
                # Add text box showing number of basins increasing and decreasing
                num_incr = (df_seas[plot] > 10).sum()
                num_decr = (df_seas[plot] < -10).sum()
                text_str = f"Inc.: {num_incr}\nDec.: {num_decr}"
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.05, text_str, transform=ax.transAxes, fontsize=fontsize - 1,
                        verticalalignment='bottom', bbox=props)

            if col == 0:  # Add labels showing simulation / isolated factor to the left-hand column
                plot_label_dict = {'co2_triffid_fix': 'S1. CLIM:\nSTOM', 'all_noLUC': 'S4.\nCLIM+CO2:\nSTOM+VEG',
                                   'CO2: STOM': 'CO2: STOM', 'CO2: STOM+VEG': 'CO2:\nSTOM+VEG', 'CLIM: VEG': 'CLIM: VEG',
                                      'CO2: STOM & CLIM+CO2: VEG': 'CO2: STOM\n&\nCLIM+CO2:\nVEG'}
                facecolor_dict = {'co2_triffid_fix': 'lightgrey', 'all_noLUC': 'lightgrey', 'CO2: STOM': 'yellowgreen',
                                  'CO2: STOM+VEG': 'yellowgreen', 'CLIM: VEG': 'yellowgreen',
                                  'CO2: STOM & CLIM+CO2: VEG': 'yellowgreen'}
                plt.text(-0.4, 0.35, plot_label_dict[plot], fontsize=fontsize, transform=ax.transAxes,
                         rotation='horizontal', bbox=dict(facecolor=facecolor_dict[plot], alpha=0.3))

    fname = 'figS6_median_wsi_by_basins_{}_reldiff_by_seasons.png'.format(period)
    plt.savefig('/home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname), dpi=300,
                bbox_inches='tight', facecolor='white')
    print('Saving plot to /home/h06/jstacey/MSc/plots/paper/final/{}'.format(fname))

    return

#plot_figS1_pop_data_map(yr_range=[2006, 2025])
#plot_figS1_pop_data_map(yr_range=[2076, 2095])
### Fig S3 plotted in main plotting file: plot_paper_figures_with_month_WSI.py
#preprocess_figS4a_calc_month_count_wsi_by_ar6_region_to_csv(period='fut', threshold=0.4)
#plot_figS4a_wsi_month_count_factors_by_ar6_hbarplots(period='fut', filter=True, fontsize=12, threshold=0.4)
#preprocess_figS4b_calc_percent_area_wsi_by_ar6_region_to_csv(period='fut', threshold=0.4)
#plot_figS4b_wsi_percent_area_by_ar6_hbarplots(period='fut', filter=True, fontsize=12, threshold=0.4)
plot_figS5_wsi_month_count_maps_by_basin(period='fut', fontsize=14)
#plot_figS6_wsi_maps_by_basin_reldiff_by_seasons(period='fut', fontsize=13)