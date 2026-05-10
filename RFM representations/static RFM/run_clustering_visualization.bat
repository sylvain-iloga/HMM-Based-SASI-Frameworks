@echo off

set INFILE=rfm_nigeria.csv
set OUTFILEpdf=rfm_nigeria.pdf
set OUTFILEhtml=rfm_nigeria.html

python clustering_visualization.py --csv_clustering %INFILE% --pdf_clustering %OUTFILEpdf% --html_clustering %OUTFILEhtml%
echo.
echo Visualization files: '%OUTFILEpdf%' and '%OUTFILEhtml%'
echo.
