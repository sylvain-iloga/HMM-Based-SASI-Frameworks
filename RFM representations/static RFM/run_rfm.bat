@echo off

set INFILE=customers_nigeria.txt
set OUTFILE=customers_nigeria.arff

python rfm.py --dataset %INFILE% --valid_dataset valid_customers_nigeria.txt --file_rfm %OUTFILE% --date_start_str 01/01/2023 --date_end_str 31/12/2024 --min_amount 12500 --T_min 150 --T_max 500 
echo.
echo RFM scores saved in '%OUTFILE%'

