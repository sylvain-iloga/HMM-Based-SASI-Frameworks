:: %1 = "C:\Program Files\Weka-3-9\weka.jar"

@echo off
setlocal enabledelayedexpansion

set KMAX=12
set INFILEagregation=agregation_nigeria.arff
set INFILElstm=lstm_nigeria.arff
set INFILEagregationlstm=agregation_lstm_nigeria.arff
set OUTFILEagregation=elbow_agregation_nigeria.csv
set OUTFILElstm=elbow_lstm_nigeria.csv
set OUTFILEagregationlstm=elbow_agregation_lstm_nigeria.csv

echo K,SSE > %OUTFILEagregation%

for /L %%K in (2,1,%KMAX%) do (

    echo K-means with K=%%K ...

    set "TMPFILE=.\weka_k_%%K.txt"

    java --add-opens java.base/java.lang=ALL-UNNAMED ^
    -cp %1 ^
    weka.clusterers.FilteredClusterer ^
    -F "weka.filters.unsupervised.attribute.Remove -R 1,17" ^
    -W weka.clusterers.SimpleKMeans ^
    -t %INFILEagregation% ^
    -- -N %%K -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10 -output-debug-info ^
    > "!TMPFILE!" 2>&1

    for /f "tokens=2 delims=:" %%B in ('findstr "squared" "!TMPFILE!"') do (
        >>"%OUTFILEagregation%" echo %%K,%%B
    )

    del "!TMPFILE!"
)
echo SSE values saved in '%OUTFILEagregation%'
echo.

echo K,SSE > %OUTFILElstm%

for /L %%K in (2,1,%KMAX%) do (

    echo K-means with K=%%K ...

    set "TMPFILE=.\weka_k_%%K.txt"

    java --add-opens java.base/java.lang=ALL-UNNAMED ^
    -cp %1 ^
    weka.clusterers.FilteredClusterer ^
    -F "weka.filters.unsupervised.attribute.Remove -R 1,17" ^
    -W weka.clusterers.SimpleKMeans ^
    -t %INFILElstm% ^
    -- -N %%K -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10 -output-debug-info ^
    > "!TMPFILE!" 2>&1

    for /f "tokens=2 delims=:" %%B in ('findstr "squared" "!TMPFILE!"') do (
        >>"%OUTFILElstm%" echo %%K,%%B
    )

    del "!TMPFILE!"
)
echo SSE values saved in '%OUTFILElstm%'
echo.

echo K,SSE > %OUTFILEagregationlstm%

for /L %%K in (2,1,%KMAX%) do (

    echo K-means with K=%%K ...

    set "TMPFILE=.\weka_k_%%K.txt"

    java --add-opens java.base/java.lang=ALL-UNNAMED ^
    -cp %1 ^
    weka.clusterers.FilteredClusterer ^
    -F "weka.filters.unsupervised.attribute.Remove -R 1,32" ^
    -W weka.clusterers.SimpleKMeans ^
    -t %INFILEagregationlstm% ^
    -- -N %%K -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10 -output-debug-info ^
    > "!TMPFILE!" 2>&1

    for /f "tokens=2 delims=:" %%B in ('findstr "squared" "!TMPFILE!"') do (
        >>"%OUTFILEagregationlstm%" echo %%K,%%B
    )

    del "!TMPFILE!"
)
echo SSE values saved in '%OUTFILEagregationlstm%'
