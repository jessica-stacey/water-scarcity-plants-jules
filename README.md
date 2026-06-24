# water-scarcity-plants-jules
## Future global water scarcity partially moderated by vegetation responses to rising CO2 (Stacey  et al., 2026)
Code for processing JULES ISIMIP2b output, calculating water scarcity and plotting paper figures.

Main files:
1. process_jules_data.py : Loads raw JULES output from ISIMIP2B suite, and processes into .nc file using Iris - this processed data is available https://doi.org/10.5281/zenodo.20826090
2. process_wsi.py : Loads and processes water supply (from processed JULES output) and water demand (from ISIMIP2B database https://data.isimip.org/search/query/amanww/tree/ISIMIP2b/OutputData/water_global/h08/hadgem2-es/)
3. plot_paper_figures.py : Code for plotting figures/tables in main paper
4. plot_paper_figs_SI.py : Code for plotting figures in supplementary information
5. common_functions_paper.py : Functions used in plotting files for data processing

Other files:
- run_setup.sh : Runs Conda environment
- .ba files used for parallel processing 
