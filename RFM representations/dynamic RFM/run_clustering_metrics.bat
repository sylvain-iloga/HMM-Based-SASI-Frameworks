@echo off

set INFILEagregation=drfm_agregation_nigeria.csv
set INFILElstm=drfm_lstm_nigeria.csv
set INFILEagregationlstm=drfm_agregation_lstm_nigeria.csv
set OUTFILE=metrics.txt

python clustering_metrics.py --csv_agregation %INFILEagregation% --csv_lstm %INFILElstm% --csv_agregation_lstm %INFILEagregationlstm% --metrics_file %OUTFILE%
echo.
echo Clustering metrics saved in '%OUTFILE%'
