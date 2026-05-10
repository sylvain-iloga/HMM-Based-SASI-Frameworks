:: %1 = Kagregation
:: %2 = Klstm
:: %3 = Kagregationlstm
:: %4 = "C:\Program Files\Weka-3-9\weka.jar"

@echo off
set Kagregation=%1
set Klstm=%2
set Kagregationlstm=%3
set INFILEagregation=agregation_nigeria.arff
set INFILElstm=lstm_nigeria.arff
set INFILEagregationlstm=agregation_lstm_nigeria.arff
set OUTFILEagregation=drfm_agregation_nigeria.csv
set OUTFILElstm=drfm_lstm_nigeria.csv
set OUTFILEagregationlstm=drfm_agregation_lstm_nigeria.csv

java --add-opens java.base/java.lang=ALL-UNNAMED ^
 -cp %4 ^
 weka.filters.MultiFilter ^
 -F "weka.filters.unsupervised.attribute.Remove -R 1,17" ^
 -F "weka.filters.unsupervised.attribute.AddCluster -W \"weka.clusterers.SimpleKMeans -N %Kagregation% -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10\"" ^
 -i %INFILEagregation% ^
 -o tmp_clustered.arff
 
python copy_column.py --original_file %INFILEagregation% --clustered_file tmp_clustered.arff --csv_file %OUTFILEagregation%

del  tmp_clustered.arff
echo DRFM-aggregation clustering results saved in '%OUTFILEagregation%'
echo.

java --add-opens java.base/java.lang=ALL-UNNAMED ^
 -cp %4 ^
 weka.filters.MultiFilter ^
 -F "weka.filters.unsupervised.attribute.Remove -R 1,17" ^
 -F "weka.filters.unsupervised.attribute.AddCluster -W \"weka.clusterers.SimpleKMeans -N %Klstm% -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10\"" ^
 -i %INFILElstm% ^
 -o tmp_clustered.arff
 
python copy_column.py --original_file %INFILElstm% --clustered_file tmp_clustered.arff --csv_file %OUTFILElstm%

del  tmp_clustered.arff
echo DRFM-lstm clustering results saved in '%OUTFILElstm%'
echo.

java --add-opens java.base/java.lang=ALL-UNNAMED ^
 -cp %4 ^
 weka.filters.MultiFilter ^
 -F "weka.filters.unsupervised.attribute.Remove -R 1,32" ^
 -F "weka.filters.unsupervised.attribute.AddCluster -W \"weka.clusterers.SimpleKMeans -N %Kagregationlstm% -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10\"" ^
 -i %INFILEagregationlstm% ^
 -o tmp_clustered.arff
 
python copy_column.py --original_file %INFILEagregationlstm% --clustered_file tmp_clustered.arff --csv_file %OUTFILEagregationlstm%

del  tmp_clustered.arff
echo DRFM-aggregation-lstm clustering results saved in '%OUTFILEagregationlstm%'
echo.
