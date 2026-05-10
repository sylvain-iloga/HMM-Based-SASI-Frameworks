@echo off

set OUTFILEcsv=customers_nigeria.csv
set OUTFILEarff=customers_nigeria.arff
set INFILE=customers_nigeria.txt

mpiexec -n 8 python customer_segmentation_para.py --dataset %INFILE% --valid_dataset valid_customers_nigeria.txt --file_mc mc_customers_nigeria.dat --csv_scores_SASI %OUTFILEcsv% --arff_scores_SASI %OUTFILEarff% --date_start_str 01/01/2023 --date_end_str 31/12/2024 --M 20 --min_amount 12500 --T_min 150 --T_max 500 --maxiter 5000 
echo.
echo Scores saved in '%OUTFILEcsv%' and '%OUTFILEarff%'
