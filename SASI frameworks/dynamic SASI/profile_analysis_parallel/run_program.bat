
:: EXAMPLE OF EXECUTION: run_program

@echo off

set OUTFILEprofileshtml=profiles_nigeria.html
set OUTFILEprofilespdf=profiles_nigeria.pdf
set OUTFILEinstabilitypdf=instability_nigeria.pdf
set OUTFILEinstabilityhtml=instability_nigeria.html
set INFILE=customers_nigeria.txt

echo.
echo =====================
echo Parallel dynamic SASI
echo =====================
echo.
mpiexec -n 8 python customer_profile_analysis_para.py --dataset %INFILE% --valid_dataset valid_customers_nigeria.txt --file_mc mc_customers_nigeria.dat --pdf_profiles %OUTFILEprofilespdf% --html_profiles %OUTFILEprofileshtml% --pdf_instability_levels %OUTFILEinstabilitypdf% --html_instability_levels %OUTFILEinstabilityhtml% --date_start_str 01/01/2023 --date_end_str 31/12/2024 --M 20 --min_amount 12500 --T_min 300 --T_max 3000 --maxiter 100 
echo.
echo Profiles saved in '%OUTFILEprofilespdf%' and '%OUTFILEprofileshtml%'
echo.
echo Instability distribution saved in '%OUTFILEinstabilitypdf%' and '%OUTFILEinstabilityhtml%'
