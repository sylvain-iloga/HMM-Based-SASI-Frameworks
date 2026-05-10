:: %1 = K
:: %2 = "C:\Program Files\Weka-3-9\weka.jar"

@echo off
set OUTFILE=rfm_nigeria.csv
set INFILE=customers_nigeria.arff

java --add-opens java.base/java.lang=ALL-UNNAMED ^
 -cp %2 ^
 weka.filters.MultiFilter ^
 -F "weka.filters.unsupervised.attribute.Remove -R 1" ^
 -F "weka.filters.unsupervised.attribute.AddCluster -W \"weka.clusterers.SimpleKMeans -N %1 -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10\"" ^
 -i %INFILE% ^
 -o tmp_clustered.arff
 
python copy_column.py --original_file %INFILE% --clustered_file tmp_clustered.arff --csv_file %OUTFILE%

del  tmp_clustered.arff
echo Clustering results saved in '%OUTFILE%'