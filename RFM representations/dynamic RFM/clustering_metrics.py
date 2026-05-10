import sys
import os
import csv
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import numpy as np
import math
import time
import argparse


# =================================================================
# 1. Reading the drfm-agregation clustering results from a CSV file
# =================================================================
def read_agregation_results(file_agregation):

    clusters = []
    matricules = []
    features_dynamic_RFM = []
    nb_transactions = []
	
    with open(file_agregation, mode='r', newline='') as f:

        reader = csv.reader(f)

        for line in reader:
            if line[0] == "matricule":
                continue
				
            matricule = line[0]
			
            Mean_R = float(line[1])
            Var_R = float(line[2])
            Slope_R = float(line[3])
            Min_R = float(line[4])
            Max_R = float(line[5])
			
            Mean_F = float(line[6])
            Var_F = float(line[7])
            Slope_F = float(line[8])
            Min_F = float(line[9])
            Max_F = float(line[10])
			
            Mean_M = float(line[11])
            Var_M = float(line[12])
            Slope_M = float(line[13])
            Min_M = float(line[14])
            Max_M = float(line[15])

            nb_transacts = int(line[16])
            cluster = int(line[17].removeprefix("c"))-1

            matricules.append(matricule)
            L = [Mean_R, Var_R, Slope_R, Min_R, Max_R, Mean_F, Var_F, Slope_F, Min_F, Max_F, Mean_M, Var_M, Slope_M, Min_M, Max_M]
            features_dynamic_RFM.append(L)
            nb_transactions.append(nb_transacts)
            clusters.append(cluster)

    labels = np.array(clusters)
    clusters_uniques, cluster_sizes = np.unique(labels, return_counts=True)
    nb_clusters = len(clusters_uniques)

    # Silhouette score
    X = np.array(features_dynamic_RFM, dtype=float)
    if nb_clusters > 1 and len(X) > nb_clusters:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = None  

    # Davies-Bouldin index
    if nb_clusters > 1:
        db_index = davies_bouldin_score(X, labels)
    else:
        db_index = None

    # Calinski-Harabasz index
    if nb_clusters > 1:
        ch_index = calinski_harabasz_score(X, labels)
    else:
        ch_index = None

    print("  1- DRFM-aggregation")
    print(f"     -silhouette: {silhouette:g}\n     -db_index: {db_index:g}\n     -ch_index: {ch_index:g}\n")

    return silhouette, db_index, ch_index
	

# ===========================================================
# 2. Reading the drfm-lstm clustering results from a CSV file
# ===========================================================
def read_lstm_results(file_lstm):

    clusters = []
    matricules = []
    features_dynamic_RFM = []
    nb_transactions = []
	
    y = np.zeros(15, dtype=float)
    with open(file_lstm, mode='r', newline='') as f:

        reader = csv.reader(f)

        for line in reader:
            if line[0] == "matricule":
                continue
				
            matricule = line[0]

            for i in range(15):
                y[i] = float(line[1 + i])
				
            nb_transacts = int(line[16])
            cluster = int(line[17].removeprefix("c"))-1

            matricules.append(matricule)
            L = [i for i in y]
            features_dynamic_RFM.append(L)
            nb_transactions.append(nb_transacts)
            clusters.append(cluster)

    labels = np.array(clusters)
    clusters_uniques, cluster_sizes = np.unique(labels, return_counts=True)
    nb_clusters = len(clusters_uniques)

    # Silhouette score
    X = np.array(features_dynamic_RFM, dtype=float)
    if nb_clusters > 1 and len(X) > nb_clusters:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = None  

    # Davies-Bouldin index
    if nb_clusters > 1:
        db_index = davies_bouldin_score(X, labels)
    else:
        db_index = None

    # Calinski-Harabasz index
    if nb_clusters > 1:
        ch_index = calinski_harabasz_score(X, labels)
    else:
        ch_index = None

    print("  2- DRFM-lstm")
    print(f"     -silhouette: {silhouette:g}\n     -db_index: {db_index:g}\n     -ch_index: {ch_index:g}\n")

    return silhouette, db_index, ch_index
	

# ======================================================================
# 3. Reading the drfm-agregation-lstm clustering results from a CSV file
# ======================================================================
def read_agregation_lstm_results(file_agregation_lstm):

    clusters = []
    matricules = []
    features_dynamic_RFM = []
    nb_transactions = []
	
    y = np.zeros(15, dtype=float)
    with open(file_agregation_lstm, mode='r', newline='') as f:

        reader = csv.reader(f)

        for line in reader:
            if line[0] == "matricule":
                continue
				
            matricule = line[0]
			
            Mean_R = float(line[1])
            Var_R = float(line[2])
            Slope_R = float(line[3])
            Min_R = float(line[4])
            Max_R = float(line[5])
			
            Mean_F = float(line[6])
            Var_F = float(line[7])
            Slope_F = float(line[8])
            Min_F = float(line[9])
            Max_F = float(line[10])
			
            Mean_M = float(line[11])
            Var_M = float(line[12])
            Slope_M = float(line[13])
            Min_M = float(line[14])
            Max_M = float(line[15])
			
            for i in range(15):
                y[i] = float(line[16 + i])
				
            nb_transacts = int(line[31])
            cluster = int(line[32].removeprefix("c"))-1

            matricules.append(matricule)
            L1 = [Mean_R, Var_R, Slope_R, Min_R, Max_R, Mean_F, Var_F, Slope_F, Min_F, Max_F, Mean_M, Var_M, Slope_M, Min_M, Max_M]
            L2 = [i for i in y]
            L = np.concatenate((L1,L2))
            features_dynamic_RFM.append(L2)
            nb_transactions.append(nb_transacts)
            clusters.append(cluster)

    labels = np.array(clusters)
    clusters_uniques, cluster_sizes = np.unique(labels, return_counts=True)
    nb_clusters = len(clusters_uniques)

    # Silhouette score
    X = np.array(features_dynamic_RFM, dtype=float)
    if nb_clusters > 1 and len(X) > nb_clusters:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = None  

    # Davies-Bouldin index
    if nb_clusters > 1:
        db_index = davies_bouldin_score(X, labels)
    else:
        db_index = None

    # Calinski-Harabasz index
    if nb_clusters > 1:
        ch_index = calinski_harabasz_score(X, labels)
    else:
        ch_index = None

    print("  3- DRFM-aggregation-lstm")
    print(f"     -silhouette: {silhouette:g}\n     -db_index: {db_index:g}\n     -ch_index: {ch_index:g}\n")

    return silhouette, db_index, ch_index
	

# ============
# Main program
# ============
def main():

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="Metrics of customer clusters")

    parser.add_argument("--csv_agregation", type=str, default="./agregation_customers.csv", help="CSV file containing the results of customer drfm-agregation clustering")
    parser.add_argument("--csv_lstm", type=str, default="./lstm_customers.csv", help="CSV file containing the results of customer drfm-lstm clustering")
    parser.add_argument("--csv_agregation_lstm", type=str, default="./agregation_lstm_customers.csv", help="CSV file containing the results of customer drfm-agregation-lstm clustering")
    parser.add_argument("--metrics_file", type=str, default="./metrics.txt", help="TXT file containing the clustering metrics")

    args = parser.parse_args()

    # Files taken as input parameters
    csv_agregation = args.csv_agregation
    csv_lstm = args.csv_lstm
    csv_agregation_lstm = args.csv_agregation_lstm
    metrics_file = args.metrics_file

    # Start of timing of the time cost
    start = time.perf_counter()

    # Reading the agregation clustering
    silhouette_agregation, db_index_agregation, ch_index_agregation = read_agregation_results(csv_agregation)

    # Reading the lstm clustering
    silhouette_lstm, db_index_lstm, ch_index_lstm = read_lstm_results(csv_lstm)

    # Reading the agregation-lstm clustering
    silhouette_agregation_lstm, db_index_agregation_lstm, ch_index_agregation_lstm = read_agregation_lstm_results(csv_agregation_lstm)

    # Saving the metrics in a file
    with open(metrics_file, "w") as f:
        f.write(f"silhouette(agregation): {silhouette_agregation:g}\n")
        f.write(f"db_index(agregation): {db_index_agregation:g}\n")
        f.write(f"ch_index(agregation): {ch_index_agregation:g}\n")
        f.write("###################################################\n")
        f.write(f"silhouette(lstm): {silhouette_lstm:g}\n")
        f.write(f"db_index(lstm): {db_index_lstm:g}\n")
        f.write(f"ch_index(lstm): {ch_index_lstm:g}\n")
        f.write("###################################################\n")
        f.write(f"silhouette(agregation-lstm): {silhouette_agregation_lstm:g}\n")
        f.write(f"db_index(agregation-lstm): {db_index_agregation_lstm:g}\n")
        f.write(f"ch_index(agregation-lstm): {ch_index_agregation_lstm:g}\n")
        f.write("###################################################\n")
     
    # End of the timing of the time cost
    end = time.perf_counter()
    print(f"Time complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	