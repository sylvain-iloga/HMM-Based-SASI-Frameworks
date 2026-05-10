:: %1 = K
:: %2 = "C:\Program Files\Weka-3-9\weka.jar"

:: EXAMPLE OF EXECUTION: run_program 6 "C:\Program Files\Weka-3-9\weka.jar"

@echo off

echo.
echo ============================================
echo 1. Sequential static SASI scores computation
echo ============================================
call run_customer_segmentation.bat
echo.

echo ===============
echo 2. ELBOW method
echo ===============
call run_elbow.bat %2
echo.

echo =====================
echo 3. K-means clustering
echo =====================
call run_kmeans.bat %1 %2
echo.

echo ===========================
echo 4. Clustering visualization
echo ===========================
call run_clustering_visualization.bat

