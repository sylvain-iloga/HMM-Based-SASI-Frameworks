:: %1 = Kagregation
:: %2 = Klstm
:: %3 = Kagregationlstm
:: %4 = "C:\Program Files\Weka-3-9\weka.jar"

:: EXAMPLE OF EXECUTION: run_program 9 6 7 "C:\Program Files\Weka-3-9\weka.jar"

@echo off

call env_tslearn\Scripts\activate
echo.
echo ===================================
echo 1. dynamic RFM features computation
echo ===================================
call run_drfm.bat
echo.
call deactivate

echo ===============
echo 2. ELBOW method
echo ===============
call run_elbow.bat %4
echo.

echo =====================
echo 3. K-means clustering
echo =====================
call run_kmeans.bat %1 %2 %3 %4
echo.

echo =====================
echo 4. Clustering metrics
echo =====================
call run_clustering_metrics.bat
