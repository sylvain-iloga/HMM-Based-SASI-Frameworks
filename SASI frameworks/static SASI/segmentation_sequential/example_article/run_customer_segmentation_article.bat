@echo off

set OUTFILEcsv=customers_article.csv
set OUTFILEarff=customers_article.arff
set INFILE=customers_article.txt

python customer_segmentation_article.py --dataset %INFILE% --valid_dataset valid_customers_article.txt --file_mc mc_customers_article.dat --file_hmm hmm_customers_article.dat --csv_scores_SASI %OUTFILEcsv% --arff_scores_SASI %OUTFILEarff% --date_start_str 01/01/2023 --date_end_str 30/04/2023 --M 10 --min_amount 10 --T_min 8 --T_max 500 --maxiter 500 
echo.
echo Scores saved in '%OUTFILEcsv%' and '%OUTFILEarff%'
