@echo off

set INFILE=kmeans_nigeria.csv
set OUTFILEpdf=kmeans_nigeria.pdf
set OUTFILEhtml=kmeans_nigeria.html

python clustering_visualization.py --csv_clustering %INFILE% --pdf_clustering %OUTFILEpdf% --html_clustering %OUTFILEhtml%
echo Visualization files: '%OUTFILEpdf%' and '%OUTFILEhtml%'
echo.
