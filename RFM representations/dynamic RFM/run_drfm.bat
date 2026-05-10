@echo off

set INFILE=customers_nigeria.txt
set OUTFILEagregation=agregation_nigeria.arff
set OUTFILElstm=lstm_nigeria.arff
set OUTFILEagregationlstm=agregation_lstm_nigeria.arff

python drfm.py --dataset %INFILE% --valid_dataset valid_customers_nigeria.txt --file_agregation %OUTFILEagregation% --file_lstm %OUTFILElstm% --file_agregation_lstm %OUTFILEagregationlstm% --date_start_str 01/01/2023 --date_end_str 31/12/2024 --dim_lstm 15 --min_amount 12500 --T_min 150 --T_max 500 
echo.
echo DRFM features saved in '%OUTFILEagregation%', '%OUTFILElstm%' and '%OUTFILEagregationlstm%'
