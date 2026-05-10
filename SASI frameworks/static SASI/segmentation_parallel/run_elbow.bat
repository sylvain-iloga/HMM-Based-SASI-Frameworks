:: %1 = "C:\Program Files\Weka-3-9\weka.jar"

@echo off
setlocal enabledelayedexpansion

set KMAX=12
set OUTFILE=elbow_sse.csv
set INFILE=customers_nigeria.arff

echo K,SSE > %OUTFILE%

for /L %%K in (2,1,%KMAX%) do (

    echo K-means with K=%%K ...

    set "TMPFILE=.\weka_k_%%K.txt"

    java --add-opens java.base/java.lang=ALL-UNNAMED ^
    -cp %1 ^
    weka.clusterers.FilteredClusterer ^
    -F "weka.filters.unsupervised.attribute.Remove -R 1,4" ^
    -W weka.clusterers.SimpleKMeans ^
    -t %INFILE% ^
    -- -N %%K -I 500 -init 2 -t1 -1.25 -t2 -1.0 -S 10 -output-debug-info ^
    > "!TMPFILE!" 2>&1

    for /f "tokens=2 delims=:" %%B in ('findstr "squared" "!TMPFILE!"') do (
        >>"%OUTFILE%" echo %%K,%%B
    )

    del "!TMPFILE!"
)

echo SSE values saved in '%OUTFILE%'