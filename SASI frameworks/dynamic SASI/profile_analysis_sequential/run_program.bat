
:: EXAMPLE OF EXECUTION: run_program

@echo off

set OUTFILEomegabar=omega_bar_nigeria.csv
set OUTFILEomega=omega_nigeria.csv
set OUTFILEinstabilitycsv=instability_nigeria.csv
set OUTFILEprofilespdf=profiles_nigeria.pdf
set OUTFILEprofileshtml=profiles_nigeria.html
set OUTFILEinstabilitypdf=instability_nigeria.pdf
set OUTFILEinstabilityhtml=instability_nigeria.html
set INFILE=customers_nigeria.txt

echo.
echo =======================
echo Sequential dynamic SASI
echo =======================
echo.
python customer_profile_analysis.py --dataset %INFILE% --valid_dataset valid_customers_nigeria.txt --file_mc mc_customers_nigeria.dat --file_hmm hmm_customers_nigeria.dat --csv_omega_bar %OUTFILEomegabar% --csv_omega %OUTFILEomega% --csv_instability_rates_and_levels %OUTFILEinstabilitycsv% --pdf_profiles %OUTFILEprofilespdf% --html_profiles %OUTFILEprofileshtml% --pdf_instability_levels %OUTFILEinstabilitypdf% --html_instability_levels %OUTFILEinstabilityhtml% --date_start_str 01/01/2023 --date_end_str 31/12/2024 --M 20 --min_amount 12500 --T_min 300 --T_max 3000 --maxiter 100  
echo.
echo Profiles saved in '%OUTFILEprofilespdf%' and '%OUTFILEprofileshtml%'
echo.
echo Instability distribution saved in '%OUTFILEinstabilitypdf%' and '%OUTFILEinstabilityhtml%'
