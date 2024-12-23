# water-scarcity-plants-jules
## Future global water  scarcity partially alleviated by vegetation responses to atmospheric CO2 and climate change (Stacey  et al., 2025) - in review
Code for processing JULES ISIMIP output, calculating water scarcity and plotting paper figures.

Main files:
1. process_jules_data.py : Loads raw JULES output from ISIMIP2B suite, and processes into .nc file using Iris
2. process_wsi.py : Loads and processes water supply (from processed JULES output) and water demand (from ISIMIP2B database)
3. plot_paper_figures.py : Code for plotting figures/tables in main paper
4. plot_paper_figs_SI.py : Code for plotting figures in supplementary information
5. common_functions_paper.py : Functions 

Other files:
- run_setup.sh : Runs Conda environment
- .ba files used for parallel processing 
