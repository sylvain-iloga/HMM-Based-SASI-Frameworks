"""

To run this program without disrupting your global Python installation, 
please create a private Python environment to install "tslearn".
Always run this program in that private Python environment, then
exit the private Python environment after running the program

1- command line to create the private environment:
     "python -m venv env_tslearn"
	 
2- command line to enter the private environment:
     "env_tslearn\Scripts\activate"
	 
3- command line to install the required packages in the private environment:
     "pip install numpy tensorflow tslearn"

4- command line to exit the private environment:
     "deactivate"

"""
import os
# Hide most logs messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable oneDNN optimizations 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import csv

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as tf_b

import time
import argparse

@dataclass
class Customer:
    matricule: str
    nb_transactions: int = 0
    transactions: List[Tuple[datetime, float]] = field(default_factory=list)


# ========================
# 1. Reading customer data
# ========================
def read_customer_data(dataset, date_start_str, date_end_str, T_min, T_max, min_amount, valid_dataset):

    customers = []
    date_start = datetime.strptime(date_start_str, "%d/%m/%Y")
    date_end = datetime.strptime(date_end_str, "%d/%m/%Y")

    with open(dataset, "r", encoding="utf-8") as f, \
         open(valid_dataset, "w", encoding="utf-8") as g:

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Separation matricule / transactions
            matricule, rest = line.split(",", 1)
            matricule = matricule.strip()

            # Transaction extraction (date,amount)
            pattern = r"\((\d{2}/\d{2}/\d{4}),([-]?\d+)\)"
            matches = re.findall(pattern, rest)

            # Initializing the list of valid transactions
            transactions_valids = []

            nb_transacts = 0
            for date_str, amount_str in matches:
                amount = float(amount_str)

                # Ignore amounts less than min_amount
                if abs(amount) < min_amount:
                    continue

                # Ignore transactions outside the analysis period
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                if date_obj < date_start or date_obj > date_end:
                    continue

                # Transaction validation
                transactions_valids.append((date_obj, amount))
                nb_transacts += 1

            # Chronological sorting of transactions
            transactions_valids.sort(key=lambda x: x[0])

            # Ignoring customers with too few transactions
            if nb_transacts < T_min:
                continue

            # Consider the recent transactions of customers with too many transactions
            if nb_transacts > T_max:
                transactions_valids = transactions_valids[-(T_max + 1):]
                nb_transacts = T_max
				
            # Skip customers that only have null amounts
            max_abs = max(abs(m) for _, m in transactions_valids)
            if max_abs == 0:
                continue				


            customer = Customer(matricule=matricule)
            customer.transactions = transactions_valids  
            customer.nb_transactions = nb_transacts
            customers.append(customer)

            # Updating the valid dataset
            g.write(customer.matricule + ",")

            for date_obj, amount in customer.transactions:
                date_str = date_obj.strftime("%d/%m/%Y")
                g.write(f"({date_str},{amount:.0f})")

            g.write("\n")

    return customers

# ===============================
# 2. Constructing the sub-windows
# ===============================
def construct_sub_windows(date_start_str, date_end_str, window_size, step):

    sub_windows = []
    date_start = datetime.strptime(date_start_str, "%d/%m/%Y")
    date_end = datetime.strptime(date_end_str, "%d/%m/%Y")
	
    window_start = date_start
    while window_start +  timedelta(days=window_size) <= date_end:
        window_end = window_start +  timedelta(days=window_size)
        sub_windows.append((window_start, window_end))
        window_start += timedelta(days=step)

    if window_start +  timedelta(days=window_size) < date_end:
        window_end = date_end
        sub_windows.append((window_start, window_end))
		
    return sub_windows

# ===============================================================
# 3. Extracts the transactions of a customer from the sub-windows
# ===============================================================
def transactions_of_sub_windows(customer, sub_windows):

    transactions = []
    windows_transactions = []
    window_index = 0
    (window_start, window_end) = sub_windows[0]
	
    for (date_obj, amount) in customer.transactions:
        while not (window_start <= date_obj <= window_end):
            windows_transactions.append(transactions)
            transactions = []
            window_index += 1
            if window_index < len(sub_windows):
                (window_start, window_end) = sub_windows[window_index]
            else:
                return windows_transactions

        transactions.append((date_obj, amount))

    while window_index < len(sub_windows):
        transactions = []
        windows_transactions.append(transactions)
        window_index += 1

    return windows_transactions

# ========================================================
# 4. Computes the dynamic RFM time series of each customer
# ========================================================
def construct_time_series(customers, sub_windows):

    R_time_series = []
    F_time_series = []
    M_time_series = []

    matricules = []
    nb_transactions = []

    for customer in customers:
        R_time_serie = []
        F_time_serie = []
        M_time_serie = []

        # Extract the transactions of the customer from the sub-windows 
        windows_transactions = transactions_of_sub_windows(customer, sub_windows)

        # Browing the transactions of each sub-window 
        (window_start, window_end) = sub_windows[0]
        last_date = window_start
        for transactions, (window_start, window_end) in zip(windows_transactions, sub_windows):
            if len(transactions) > 0:    
                # Non-empty sub-window 
                dates = [date_obj for (date_obj, _) in transactions]
                amounts = [amount for (_, amount) in transactions]

                # R = recency 
                # days from the last transaction to the end of the current window
                last_date = dates[-1]
                R = (window_end - last_date).days

                # F = frequency
                # number of transactions
                F = len(transactions)

                # M = Monetary
                # Volume
                M = sum([abs(amount) for amount in amounts])
					
            else:    
                # Empty sub-window 
                R = (window_end - last_date).days
                F = 0
                M = 0

            # Updating the time series of the current customer
            R_time_serie.append(R)
            F_time_serie.append(F)
            M_time_serie.append(M)

        # Recording the time series of the current customer
        R_time_series.append(R_time_serie)
        F_time_series.append(F_time_serie)
        M_time_series.append(M_time_serie)

        # Recording the matricule and the number of transactions of the current customer
        matricules.append(customer.matricule)
        nb_transactions.append(customer.nb_transactions)
			
    return matricules, nb_transactions, R_time_series, F_time_series, M_time_series


# ====================================
# 5. Calculates the agregated features
# ====================================
def agregate_features(time_serie):

    time_serie = np.array(time_serie, dtype=float)
    n = len(time_serie)

    # Mean and Variance
    Mean, Var = np.mean(time_serie), np.var(time_serie)

    # Linear regression slope (least squares)
    # A = covariance(t, time_serie) / variance(t)
    t = np.arange(n)
    if n > 1:
        Slope = np.cov(t, time_serie, bias=True)[0, 1] / np.var(t)
    else:
        Slope = 0.0  

    # Min and Max 
    Min, Max = np.min(time_serie), np.max(time_serie)

    return Mean, Var, Slope, Min, Max

# ===================================================
# 6. Saves dynamic RFM-agregate features in ARFF file
# ===================================================
def save_agregate_features(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, file_agregation):

    # Computing R-agregate features for the current customer
    R_agregate = []
    for R_time_serie in R_time_series:
        Mean, Var, Slope, Min, Max = agregate_features(R_time_serie)
        R_agregate.append((Mean, Var, Slope, Min, Max))

    # Computing F-agregate features for the current customer
    F_agregate = []
    for F_time_serie in F_time_series:
        Mean, Var, Slope, Min, Max = agregate_features(F_time_serie)
        F_agregate.append((Mean, Var, Slope, Min, Max))

    # Computing M-agregate features for the current customer
    M_agregate = []
    for M_time_serie in M_time_series:
        Mean, Var, Slope, Min, Max = agregate_features(M_time_serie)
        M_agregate.append((Mean, Var, Slope, Min, Max))

    # Saving dynamic RFM agregate features in ARFF file
    with open(file_agregation, "w", encoding="utf-8") as f:

        f.write("@RELATION agregation\n\n")

        f.write("@ATTRIBUTE matricule STRING\n")
        f.write("@ATTRIBUTE Mean_R REAL\n")
        f.write("@ATTRIBUTE Var_R REAL\n")
        f.write("@ATTRIBUTE Slope_R REAL\n")
        f.write("@ATTRIBUTE Min_R REAL\n")
        f.write("@ATTRIBUTE Max_R REAL\n")
        f.write("@ATTRIBUTE Mean_F REAL\n")
        f.write("@ATTRIBUTE Var_F REAL\n")
        f.write("@ATTRIBUTE Slope_F REAL\n")
        f.write("@ATTRIBUTE Min_F REAL\n")
        f.write("@ATTRIBUTE Max_F REAL\n")
        f.write("@ATTRIBUTE Mean_M REAL\n")
        f.write("@ATTRIBUTE Var_M REAL\n")
        f.write("@ATTRIBUTE Slope_M REAL\n")
        f.write("@ATTRIBUTE Min_M REAL\n")
        f.write("@ATTRIBUTE Max_M REAL\n")
        f.write("@ATTRIBUTE nb_transactions INTEGER\n\n")

        f.write("@DATA\n")

        for matricule, r_val, f_val, m_val, t in zip(matricules, R_agregate, F_agregate, M_agregate, nb_transactions):
            f.write(f"\"{matricule}\",")	
            (Mean, Var, Slope, Min, Max) = r_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	
            (Mean, Var, Slope, Min, Max) = f_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	
            (Mean, Var, Slope, Min, Max) = m_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	
            f.write(f"{int(t)}\n")	
			
    return R_agregate, F_agregate, M_agregate

# ======================================================================
# 7. Saves dynamic RFM-LSTM and RFM-agregate-LSTM features in ARFF files
# ======================================================================
def save_lstm_agregate(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, dim_lstm, R_agregate, F_agregate, M_agregate, file_lstm, file_agregation_lstm, epochs=500, batch_size=32):

    # the number of customers
    T = len(matricules)

    # Build multivariate temporal series
    X = []
    for customer in range(T):
        series = np.column_stack((R_time_series[customer], F_time_series[customer], M_time_series[customer]))
        X.append(series)

    # Data normalization
    X = to_time_series_dataset(X)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    n_timesteps = X.shape[1]
    n_features = X.shape[2]

    # Define the LSTM encoder
    inputs = Input(shape=(n_timesteps, n_features))
    encoded = LSTM(dim_lstm, activation='tanh')(inputs)
    encoded = LayerNormalization()(encoded)
    encoded = Lambda(lambda x: tf_b.l2_normalize(x, axis=1))(encoded)
	
    # Define the LSTM decoder
    decoded = RepeatVector(n_timesteps)(encoded)
    decoded = LSTM(n_features, activation='tanh', return_sequences=True)(decoded)

    # Pointer to the encoder 
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)  

    # EarlyStopping
    early_stop = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4, restore_best_weights=True)

    # Training the model using the Adam optimizer
    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    # Generate the T vectors, each having dim_lstm components
    lstm_vectors = encoder.predict(X)  

    # Save results in a ARFF files
    with open(file_agregation_lstm, "w", encoding="utf-8") as f:
        # Header ARFF
        f.write("@RELATION agregation_lstm\n\n")
        f.write("@ATTRIBUTE matricule STRING\n")

        f.write("@ATTRIBUTE Mean_R REAL\n")
        f.write("@ATTRIBUTE Var_R REAL\n")
        f.write("@ATTRIBUTE Slope_R REAL\n")
        f.write("@ATTRIBUTE Min_R REAL\n")
        f.write("@ATTRIBUTE Max_R REAL\n")
        f.write("@ATTRIBUTE Mean_F REAL\n")
        f.write("@ATTRIBUTE Var_F REAL\n")
        f.write("@ATTRIBUTE Slope_F REAL\n")
        f.write("@ATTRIBUTE Min_F REAL\n")
        f.write("@ATTRIBUTE Max_F REAL\n")
        f.write("@ATTRIBUTE Mean_M REAL\n")
        f.write("@ATTRIBUTE Var_M REAL\n")
        f.write("@ATTRIBUTE Slope_M REAL\n")
        f.write("@ATTRIBUTE Min_M REAL\n")
        f.write("@ATTRIBUTE Max_M REAL\n")

        for i in range(dim_lstm):
            f.write(f"@ATTRIBUTE lstm_{i+1} REAL\n")
        f.write("@ATTRIBUTE nb_transactions INTEGER\n")
		
        f.write("\n@DATA\n")

        # Data
        for matricule, r_val, f_val, m_val, lstm_vector, t in zip(matricules, R_agregate, F_agregate, M_agregate, lstm_vectors, nb_transactions):
            f.write(f"\"{matricule}\",")	

            (Mean, Var, Slope, Min, Max) = r_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	
            (Mean, Var, Slope, Min, Max) = f_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	
            (Mean, Var, Slope, Min, Max) = m_val
            f.write(f"{Mean:g},{Var:g},{Slope:g},{Min:g},{Max:g},")	

            line = [f"{component:g}" for component in lstm_vector]
            f.write(",".join(line) + ",")
            f.write(f"{int(t)}\n")	


    with open(file_lstm, "w", encoding="utf-8") as g:
        # Header ARFF
        g.write("@RELATION lstm\n\n")
        g.write("@ATTRIBUTE matricule STRING\n")
        for i in range(dim_lstm):
            g.write(f"@ATTRIBUTE lstm_{i+1} NUMERIC\n")
        g.write("@ATTRIBUTE nb_transactions INTEGER\n")

        g.write("\n@DATA\n")

        # Data
        for matricule, lstm_vector, t in zip(matricules, lstm_vectors, nb_transactions):
            g.write(f"\"{matricule}\",")	
            line = [f"{component:g}" for component in lstm_vector]
            g.write(",".join(line) + ",")
            g.write(f"{int(t)}\n")	

    return lstm_vectors

# ==========================================================
# 8. Save Kmeans-DTW results in ARFF file for a particular K
# ==========================================================
def kmeans_dtw(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, K, file_kmeans_dtw):

    T = len(matricules)

    # Build multivariate temporal series
    X = []
    for customer in range(T):
        series = np.column_stack((R_time_series[customer], F_time_series[customer], M_time_series[customer]))
        X.append(series)

    # Data normalization because DTW is sensitive to the amplitude
    X = to_time_series_dataset(X)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)
	
    # Run kmeans-dtw
    model = TimeSeriesKMeans(
        n_clusters=K,
        metric="dtw",
        max_iter=20,
        random_state=0,
        metric_params={"sakoe_chiba_radius": 5},
        n_init=2
    )


    # Clustering step 
    clusters = model.fit_predict(X)
    labels = np.array(clusters)
    clusters_uniques = np.unique(labels)

    
    # Save clustering results in a ARFF file
    with open(file_kmeans_dtw, "w", newline="", encoding="utf-8") as f:

        # Relation name
        f.write(f"@RELATION kmeans_dtw_{K}\n\n")

        # Attributes
        f.write(f"@ATTRIBUTE matricule STRING\n")
        for i in range(len(R_time_series[customer])):
            f.write(f"@ATTRIBUTE R{i+1} INTEGER\n")
        for i in range(len(F_time_series[customer])):
            f.write(f"@ATTRIBUTE F{i+1} INTEGER\n")
        for i in range(len(M_time_series[customer])):
            f.write(f"@ATTRIBUTE M{i+1} REAL\n")
        f.write("@ATTRIBUTE nb_transactions INTEGER\n")
        f.write("@ATTRIBUTE Cluster {")
        for i in range(len(clusters_uniques)-1):
            f.write(f"cluster{i},")
        f.write(f"cluster{len(clusters_uniques)-1}" + "}\n\n")

        # Data
        f.write("@DATA\n")

        for matricule, c, customer, t in zip(matricules, clusters, range(T), nb_transactions):
            f.write(f"\"{matricule}\",")
            for value in R_time_series[customer]:
                f.write(f"{value:g},")
            for value in F_time_series[customer]:
                f.write(f"{value:g},")
            for value in M_time_series[customer]:
                f.write(f"{value:g},")
            f.write(f"{int(t)},")	
            f.write(f"cluster{c}\n")

    # SSE 
    sse = model.inertia_

    # Silhouette 
    D = cdist_dtw(X)
    silhouette = silhouette_score(D, clusters, metric="precomputed")

    # Davies-Bouldin
    X_flat = X.reshape((X.shape[0], -1))
    db_index = davies_bouldin_score(X_flat, clusters)

    # Calinski-Harabasz
    ch_index = calinski_harabasz_score(X_flat, clusters)

    return sse, silhouette, db_index, ch_index

# =========================================
# 9. Run Kmeans-DTW for several values of K
# =========================================
def kmeans_dtw_multi(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, Kmax):

    sses = []
    silhouettes = []
    db_indexes = []
    ch_indexes = []

    # Creating the sub-folder
    os.makedirs("./kmeans_dtw", exist_ok=True)
	
    # Runs kmeans-dtw for k = 2...Kmax
    print("\n############### kmeans-dtw ###############\n")
    for k in range(2,Kmax+1):
        file_clustering = f"./kmeans_dtw/kmeans_dtw_{k}.arff"  
        sse, silhouette, db_index, ch_index = kmeans_dtw(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, k, file_clustering)	
        sses.append(sse)
        silhouettes.append(silhouette)	
        db_indexes.append(db_index)	
        ch_indexes.append(ch_index)	
        print(f"k:{k}, silhouette: {silhouette:g}, db_index: {db_index:g}, ch_index: {ch_index:g}\n")

    # Save the resulting metrics in a CSV file
    with open("./metrics_kmeans_dtw.csv", "w") as f:
        f.write("K,sse,silhouette,db_index,ch_index\n")
        for k, sse, silhouette, db_index, ch_index in zip(range(2,Kmax+1), sses, silhouettes, db_indexes, ch_indexes): 
            f.write(f"{k},{sse:g},{silhouette:g},{db_index:g},{ch_index:g}\n")

			
# ============
# Main program
# ============
def main():

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="Customer segmentation for a bank using dynamic RFM")

    parser.add_argument("--dataset", type=str, default="./customers.txt", help="Customer bank transaction file")
    parser.add_argument("--valid_dataset", type=str, default="./valid_customers.txt", help="Valid customer transaction file")
    parser.add_argument("--file_agregation", type=str, default="./agregation.arff", help="ARFF file containing the customers features using agregation")
    parser.add_argument("--file_lstm", type=str, default="./lstm.arff", help="ARFF file containing the customers features using LSTM autoencoder")
    parser.add_argument("--file_agregation_lstm", type=str, default="./agregation_lstm.arff", help="ARFF file containing the customers features using agregation-LSTM autoencoder")
    parser.add_argument("--date_start_str", type=str, default="01/01/2023", help="Start date of the customer analysis period")
    parser.add_argument("--date_end_str", type=str, default="31/12/2023", help="End date of the customer analysis period")
    parser.add_argument("--dim_lstm", type=int, default=15, help="Number of LSTM latent features.")
    parser.add_argument("--min_amount", type=int, default=12500, help="Minimum amount for a valid transaction.")
    parser.add_argument("--T_min", type=int, default=150, help="Minimum number of valid transactions.")
    parser.add_argument("--T_max", type=int, default=500, help="Maximum number of valid recent transactions.")

    args = parser.parse_args()

    # Files taken as input parameters
    dataset = args.dataset
    valid_dataset = args.valid_dataset
    file_agregation = args.file_agregation
    file_lstm = args.file_lstm
    file_agregation_lstm = args.file_agregation_lstm

    # Verification of the analysis duration 
    date_start_str = args.date_start_str
    date_end_str = args.date_end_str
    date_start = datetime.strptime(date_start_str, "%d/%m/%Y")
    date_end = datetime.strptime(date_end_str, "%d/%m/%Y")

    # always have (date_start <= date_end) 
    if date_start > date_end:
        date_start, date_end = date_end, date_start

    # minimum duration of around 6 months 
    nb_days_analysis = (date_end - date_start).days
    if nb_days_analysis < 181:
        print(f"Analysis time too short ({nb_days_analysis} days).")
        print(f"Minimum analysis time: 181 days.")
        return 

    date_start_str = date_start.strftime("%d/%m/%Y")
    date_end_str = date_end.strftime("%d/%m/%Y")
    print(f"\nAnalysis from {date_start_str} to {date_end_str}")

    # min_amount must be positive
    min_amount = args.min_amount
    if min_amount < 0:
        min_amount = 12500

    # T_min between 150 and 500
    T_min = args.T_min
    if T_min < 150:
        T_min = 150
    elif T_min > 500:
        T_min = 500
		
    # T_max between 150 and 500
    T_max = args.T_max
    if T_max < 150:
        T_max = 150
    elif T_max > 500:
        T_max = 500

    # always have (T_min <= T_max) 
    if T_min > T_max:
        T_min, T_max = T_max, T_min
		
    # dim_lstm between 5 and 20
    dim_lstm = args.dim_lstm
    if dim_lstm < 5:
        dim_lstm = 5
    elif dim_lstm > 20:
        dim_lstm = 20

    # Other parameters
    window_size = 180
    step = 30
    Kmax = 12
	
    # Start of timing of the time cost
    start = time.perf_counter()
    
    # Constructing the windows
    sub_windows = construct_sub_windows(date_start_str, date_end_str, window_size, step)

    # Reading customer data
    customers = read_customer_data(dataset, date_start_str, date_end_str, T_min, T_max, min_amount, valid_dataset)
    print(f"\nNumber of valid customers :{len(customers)}")
    print(f"{len(sub_windows)} windows.\n")

    # Compute the dynamic RFM time series of each customer
    print("    1- Computing the DRFM raw time-series")
    matricules, nb_transactions, R_time_series, F_time_series, M_time_series = construct_time_series(customers, sub_windows)

    # Save dynamic RFM-agregate features in ARFF file
    print("    2- Computing the DRFM-aggregation features")
    R_agregate, F_agregate, M_agregate = save_agregate_features(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, file_agregation)

    # Save dynamic RFM-LSTM and RFM-agregate-LSTM features in ARFF file
    print("    3- Computing the DRFM-lstm and DRFM-lstm-aggregation features")
    lstm_vectors = save_lstm_agregate(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, dim_lstm, R_agregate, F_agregate, M_agregate, file_lstm, file_agregation_lstm)

    # Run kmeans-dtw for K = 2..Kmax, results are saved in ARFF files 
    # WARNING: This function is significantly time consuming (around 3h30 on the experimental computer) 
    print(f"    4- Running kmeans-DTW for K = 2..{Kmax}")
    kmeans_dtw_multi(matricules, nb_transactions, R_time_series, F_time_series, M_time_series, Kmax)

    # End of the timing of the time cost
    end = time.perf_counter()
    print(f"Time complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	