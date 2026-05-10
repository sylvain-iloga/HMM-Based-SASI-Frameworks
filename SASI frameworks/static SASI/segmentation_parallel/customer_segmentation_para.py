from mpi4py import MPI
import sys
import re
import os
import csv
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import math
import copy
# pomegranate version 0.14.8
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution
import time
import argparse

# ==========
# A customer
# ==========
@dataclass
class Customer:
    matricule: str
    nb_transactions: int = 0
    amount_max: float = 0.0
    transactions: List[Tuple[datetime, float]] = field(default_factory=list)
    transactions_norm: List[Tuple[datetime, float]] = field(default_factory=list)
    markov_chain: List[Tuple[int, int]] = field(default_factory=list)


# ================================
# 0. Customer partitioning
# ================================
def partition_customers(customers, P):
    n = len(customers)
    size = n // P
    blocs = []

    for i in range(P):
        start = i * size
        if i == P-1:
            end = n
        else:
            end = (i+1) * size
        blocs.append(customers[start:end])

    return blocs
	
# ===================================
# 1. Symbol associated with an amount
# ===================================
def symbol_amount(amount, M):

    if amount == 0.0:
        y = 0
    elif amount > 0.0:
        for k in range(1, M + 1):
            if (100.0 / M) * (k - 1) < amount <= (100.0 / M) * k:
                y = k
                break
    elif amount < 0.0:
        for k in range(1, M + 1):
            if (-100.0 / M) * k <= amount < (-100.0 / M) * (k - 1):
                y = k+M
                break
    return y 

# =========================================
# 2. State associated with a number of days
# =========================================
def state_number_of_days(nb_days):

    if nb_days == 0:
        # the same day
        x = 0
    elif 1 <= nb_days <= 3:
        # between 1 day and 3 days
        x = 1
    elif 4 <= nb_days <= 7:
        # between 4 days and 1 week
        x = 2
    elif 8 <= nb_days <= 30:
        # between 1 week and 1 month
        x = 3
    elif 31 <= nb_days <= 60:
        # between 1 and 2 months
        x = 4
    elif 61 <= nb_days <= 180:
        # between 2 months and 6 months
        x = 5
    else:
        # beyond 6 months
        x = 6

    return x 

# ========================
# 3. Reading customer data
# ========================
def read_customer_data(dataset, date_start_str, date_end_str, file_mc, valid_dataset, M, T_min, T_max, min_amount):
	
    # Converting dates from text to datetime objects
    date_start = datetime.strptime(date_start_str, "%d/%m/%Y")
    date_end = datetime.strptime(date_end_str, "%d/%m/%Y")

    # Initializing the list of customers
    customers = []

    with open(dataset, "r", encoding="utf-8") as f, \
         open(file_mc, "w", encoding="utf-8") as g, \
         open(valid_dataset, "w", encoding="utf-8") as h:

        for line in f:
            line = line.strip()

            # Skip empty lines
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
            transactions_valids.append((date_start, 0.0))

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

            # Finalizing the list of transactions
            transactions_valids.append((date_end, 0.0))

            # Chronological sorting of transactions
            transactions_valids.sort(key=lambda x: x[0])

            # Ignoring customers with too few transactions
            if nb_transacts < T_min:
                continue

            # Consider the recent transactions of customers with too many transactions
            if nb_transacts > T_max:
                transactions_valids = transactions_valids[-(T_max + 1):]
                nb_transacts = T_max

            customer = Customer(matricule=matricule)
            customer.transactions = transactions_valids  
            customer.nb_transactions = nb_transacts
				
            # Skip customers that only have null amounts
            max_abs = max(abs(m) for _, m in transactions_valids)
            if max_abs == 0:
                continue				

            # Normalization
            customer.amount_max = max_abs

            for date_obj, amount in transactions_valids:
                amount_norm = (amount / customer.amount_max) * 100.0
                customer.transactions_norm.append((date_obj, amount_norm))

            # Construction of the Markov chain
            prev_date = date_start

            for i, ((date_obj, amount),
                    (_, amount_norm)) in enumerate(zip(customer.transactions,
                                                        customer.transactions_norm)):

                # start with the customer's first effective transaction
                if i == 0:
                    continue
					
                # Calculating state x 
                nb_days = (date_obj - prev_date).days
                x = state_number_of_days(nb_days)
                prev_date = date_obj

                # Calculating symbol y 
                y = symbol_amount(amount_norm, M)

                customer.markov_chain.append((x, y))

            # Saving the Markov chain in a file
            g.write(customer.matricule + ",")

            for x, y in customer.markov_chain:
                g.write(f"({x},{y})")

            g.write("\n")

            customers.append(customer)

            # Updating the valid dataset
            h.write(customer.matricule + ",")

            for date_obj, amount in customer.transactions:
                if (date_obj == date_start or date_obj == date_end) and amount == 0.0: 
                    continue
                date_str = date_obj.strftime("%d/%m/%Y")
                h.write(f"({date_str},{amount:.0f})")

            h.write("\n")

    return customers

# ===================================
# 4. Readjusting the rows of a matrix
# ===================================
def readjust_rows(MAT):
    for i in range(MAT.shape[0]):
        s = MAT[i, :].sum()
        if s < 1.0:
            MAT[i, :] += (1.0 - s) / MAT.shape[1]
    return MAT


# =========================================
# 5. Readjusting the components of a vector
# =========================================
def readjust_vect(v):
    s = v.sum()
    if s < 1.0:
        v += (1.0 - s) / len(v)
    return v

# ===========================================
# 6. Construction of a customer's initial HMM
# ===========================================
def initial_hmm(file_mc, N, M, epsilon, minfloat):

    # Initialisation des variables
    transit = np.zeros((N, N), dtype=float)
    observe = np.zeros((N, M), dtype=float)
    start = np.zeros(N, dtype=float)
	
    # number of Markov chains in the file
    L = 0  
    # List of sequences of symbols
    sequences = []  

    pair_pattern = re.compile(r'\((\d+)\s*,\s*(\d+)\)')

    with open(file_mc, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs = pair_pattern.findall(line)
            if not pairs:
                continue
            L += 1
            states = [int(s) for s, _ in pairs]
            symbols = [int(o) for _, o in pairs]
            sequences.append(symbols)

            s0 = states[0]
            start[s0] += 1

            for t in range(len(states)):
                sta, symb = states[t], symbols[t]
                observe[sta, symb] += 1
                if t < len(states) - 1:
                    transit[sta, states[t+1]] += 1

    # Initializing HMM parameters
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, M), dtype=float)
    pi = np.zeros(N, dtype=float)

    # Calculation of HMM parameters
    transit_sum = transit.sum(axis=1)
    observe_sum = observe.sum(axis=1)

    for i in range(N):
        A[i, :] = transit[i, :] / (transit_sum[i] + epsilon)
    for j in range(N):
        B[j, :] = observe[j, :] / (observe_sum[j] + epsilon)
    pi[:] = start / (L + epsilon)

    # HMM parameter readjustment
    A = readjust_rows(A)
    B = readjust_rows(B)
    pi = readjust_vect(pi)

    # Safety normalization of HMM parameters
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    pi = pi / np.sum(pi)

    # HMM parameter thresholding for improved readability
    A_threshold = np.where(A < minfloat, 0.0, A)
    B_threshold = np.where(B < minfloat, 0.0, B)
    pi_threshold = np.where(pi < minfloat, 0.0, pi)

    return A_threshold, B_threshold, pi_threshold, sequences

# ====================================================
# 7. Construction of the initial HMMs of the customers
# ====================================================
def initial_hmms_customers(num_proc, customers, N, M, epsilon, minfloat):

    initial_hmms = []
	
    for customer in customers:
        mc = ''.join(f'({x},{y})' for (x,y) in customer.markov_chain) + "\n" 
        with open(f"./proc_{num_proc}/tmp.txt", "w", encoding="utf-8") as g:
            g.write(mc)  
				
        A, B, pi, sequences = initial_hmm(f"./proc_{num_proc}/tmp.txt", N, M, epsilon, minfloat)
        initial_hmms.append((A, B, pi, sequences))
			
    os.remove(f"./proc_{num_proc}/tmp.txt")

    return initial_hmms

# =====================================================
# 8. Constructing an HMM object with pomegranate 0.14.8
# =====================================================
def build_hmm_pomegranate(A, B, pi):
    # Extraction of N and M
    N = B.shape[0]
    M = B.shape[1]

    # Construction of the HMM object  
    distributions = [DiscreteDistribution({k: float(B[i, k]) for k in range(M)})
                     for i in range(N)]
    hmm = HiddenMarkovModel.from_matrix(A, distributions, pi)
    hmm.bake()

    return hmm

# ===============================================================================
# 9. Extracting parameters (A, B, pi) from an HMM object using pomegranate 0.14.8
# ===============================================================================
def parameters_hmm_pomegranate(hmm, minfloat):

    # 1. Preparation for extraction

    # Complete matrix including start and end states
    A_tmp = hmm.dense_transition_matrix()

    # States start and end
    start = hmm.start_index
    end = hmm.end_index

    # Indexes of the true states
    states = [i for i in range(len(A_tmp)) 
                     if i not in (start, end)]

    # 2. Extraction of A by Cartesian product 
    A = A_tmp[np.ix_(states, states)]

    # 3. Extracting B from the dictionary
    B_tmp = [s.distribution.parameters
             for s in hmm.states
             if s.distribution is not None]

    B = np.array([[row[0][k] for k in sorted(row[0])]
                  for row in B_tmp], dtype=float)

    # 4. Extraction of pi
    pi = A_tmp[start, states]
		
    # 5. Threshold for improved readability
    A_final = np.where(A < minfloat, 0.0, A)
    B_final = np.where(B < minfloat, 0.0, B)
    pi_final = np.where(pi < minfloat, 0.0, pi)

    return A_final, B_final, pi_final
	
# ==================================================
# 10. Training an HMM object with pomegranate 0.14.8
# ==================================================
def train_hmm_pomegranate(hmm_init, sequences, maxiter, minfloat):
    # Baum-Welch algorithm
    hmm_final = hmm_init.fit(sequences, algorithm='baum-welch', max_iterations=maxiter, verbose=False)
    hmm_final.bake()

    # Extracting the final parameters
    A_bar, B_bar, pi_bar = parameters_hmm_pomegranate(hmm_final, minfloat)

    return A_bar, B_bar, pi_bar
	
# ================================
# 11. Vector associated with a HMM
# ================================
def vector_hmm(A, B, minfloat, r, tolerance):

    # 1. Calculating the stationary distribution

    # Calculating A^(r)  
    A_power = np.copy(A)
    for i in range(1, r):
        A_next = A_power @ A
        if np.allclose(A_next, A_power, atol=tolerance):
            break
        A_power = A_next

    # here, phi = first line of A^(r)
    phi_tmp = A_power[0, :]
	# safety normalization
    phi_tmp = phi_tmp / np.sum(phi_tmp)  
    # Threshold for improved readability
    phi = np.where(phi_tmp < minfloat, 0.0, phi_tmp)

    # 2. Calculation of the vector phi_bar
    phi_bar_tmp = phi @ B
    # Threshold for improved readability
    phi_bar = np.where(phi_bar_tmp < minfloat, 0.0, phi_bar_tmp)

    return phi, phi_bar
	
# ==========================================
# 12. Saving a customer's HMM in a text file
# ==========================================
def save_hmm(matricule, A, B, pi, phi, phi_bar, file_hmm):
    separator = "========================================================"
    separator_customers = "########################################################"

    with open(file_hmm, "a") as f:

        # 1. Matricule
        f.write(separator_customers + "\n")
        f.write(matricule + "\n")
        f.write(separator_customers + "\n")

        # 2. Matrix A
        f.write("A:\n")
        for line in A:
            values = ",".join("%g" % val for val in line)
            f.write(values + "\n")
        f.write(separator + "\n")

        # 3. Matrix B
        f.write("B:\n")
        for line in B:
            values = ",".join("%g" % val for val in line)
            f.write(values + "\n")
        f.write(separator + "\n")

        # 4. Vector pi
        f.write("pi:\n")
        values = ",".join("%g" % val for val in pi)
        f.write(values + "\n")
        f.write(separator + "\n")

        # 5. Stationary distribution
        f.write("phi:\n")
        np.savetxt(f, phi.reshape(1, -1), delimiter=',', fmt="%g")
        f.write(separator + "\n")

        # 6. Vector phi_bar
        f.write("phi_bar:\n")
        np.savetxt(f, phi_bar.reshape(1, -1), delimiter=',', fmt="%g")
        f.write("\n")

# ===========================
# 13. Customers' HMM training
# ===========================
def train_hmms_customers(customers, initial_hmms, maxiter, minfloat, r, tolerance, file_hmm):
    
    features = []
    for (A_init, B_init, pi_init, sequences), customer in zip(initial_hmms, customers):  
        hmm_init = build_hmm_pomegranate(A_init, B_init, pi_init)
        A, B, pi = train_hmm_pomegranate(hmm_init, sequences, maxiter, minfloat)
        phi, phi_bar = vector_hmm(A, B, minfloat, r, tolerance)
        features.append((phi, phi_bar))
        save_hmm(customer.matricule, A, B, pi, phi, phi_bar, file_hmm)

    return features

# ========================================
# 14. Quadratic coefficients for SA and SI
#     SA = Sequential Amounts
#     SI = Sequential Intensity
# ========================================
def coefficients_SASI(nb_states, M):

    coefs_SI = np.zeros(nb_states, dtype=float)
    for k in range(nb_states):
        coefs_SI[k] = (nb_states - k)**2

    coefs_SA = np.zeros(2*M+1, dtype=float)
    coefs_SA[0] = 0.0
    for k in range(1, M+1):
        coefs_SA[k] = k**2
        coefs_SA[k + M] = -k**2

    return coefs_SI, coefs_SA


# ===============================================
# 16. Calculating the scores for F and I
#     F = net Flow (balance deposits-withdrawals)
#     I = transactional Intensity
# ===============================================
def compute_scores_SASI(customers, features, coefs_SI, coefs_SA, T_max, file_scores_SASI):

    nb_symbols = len(coefs_SA)
    nb_states = len(coefs_SI)
    list_x = []
    list_y = []
	
    for (phi, phi_bar), customer in zip(features, customers):  

        # Calculating x
        x = 0.0
        for k in range(nb_symbols):
            x = x + coefs_SA[k]*phi_bar[k]
		
        # Weighting x by the maximum amount
        x *= customer.amount_max 
		
        # Reduction of asymmetry and robustness to outliers
        x = math.copysign(math.log1p(abs(x)), x)
		
        # Updating list_x
        list_x.append(x)

        # Calculating y
        y = 0.0
        for k in range(nb_states):
            y = y + coefs_SI[k]*phi[k]
		
        # Weighting y by the number of transactions
        y *= customer.nb_transactions 

        # Reduction of asymmetry and robustness to outliers
        y = math.log1p(y)
		
        # Updating list_y
        list_y.append(y)

    # Normalization	of list_y in [0,100]
    y_max = (nb_states**2)*T_max
    y_max = math.log1p(y_max)
    list_y = [100.0 * (y / y_max) for y in list_y]
	
    # Saving the scores 	
    scores_SASI = []

    with open(file_scores_SASI, "w") as f:
        f.write("matricule,x,y\n")
        for customer, x, y  in zip(customers, list_x, list_y):  
            scores_SASI.append((x, y))
            f.write(customer.matricule + ",")
            f.write(f"{x:g},")
            f.write(f"{y:g}\n")

    return scores_SASI


# =====================================================
# 16. Building the ARFF file associated with customers
#     for automatic clustering in Weka
# =====================================================
def build_arff(customers, scores_SASI, file_arff):

    with open(file_arff, "w", encoding="utf-8") as f:

        # ARFF file header
        f.write("@RELATION customers\n")
        f.write("@ATTRIBUTE matricule STRING\n")
        f.write("@ATTRIBUTE x REAL\n")
        f.write("@ATTRIBUTE y REAL\n")
        f.write("@ATTRIBUTE nb_transactions INTEGER\n\n")

        # Data
        f.write("@DATA\n")
        for customer, (x, y) in zip(customers, scores_SASI):

            line = f"\"{customer.matricule}\",{x:g},{y:g},{customer.nb_transactions}\n"
            f.write(line)

# ============
# Main program
# ============
def main():  

    # Initialization of MPI variables
    comm = MPI.COMM_WORLD
    num_proc = comm.Get_rank()
    nb_procs = comm.Get_size()

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="Parallel customer segmentation using HMMs")

    parser.add_argument("--dataset", type=str, default="./customers.txt", help="Customer transaction file")
    parser.add_argument("--valid_dataset", type=str, default="./valid_customers.txt", help="Valid customer transaction file")
    parser.add_argument("--file_mc", type=str, default="./mc_customers.dat", help="File containing the Markov chains of the customers")
    parser.add_argument("--csv_scores_SASI", type=str, default="./scores_SASI_customers.csv", help="CSV file containing the scores (x,y) of the customers")
    parser.add_argument("--arff_scores_SASI", type=str, default="./customers.arff", help="ARFF file containing the scores (x,y) of the customers")
    parser.add_argument("--date_start_str", type=str, default="01/01/2023", help="Start date of the customer analysis period")
    parser.add_argument("--date_end_str", type=str, default="31/12/2023", help="End date of the customer analysis period")
    parser.add_argument("--M", type=int, default=20, help="The number of symbols is (2*M+1).")
    parser.add_argument("--min_amount", type=int, default=12500, help="Minimum amount for a valid transaction.")
    parser.add_argument("--T_min", type=int, default=150, help="Minimum number of valid transactions.")
    parser.add_argument("--T_max", type=int, default=500, help="Maximum number of valid recent transactions.")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of iterations for the Baum-Welch algorithm.")

    args = parser.parse_args()

    # Files taken as input parameters
    dataset = args.dataset
    valid_dataset = args.valid_dataset
    file_mc = args.file_mc
    csv_scores_SASI = args.csv_scores_SASI
    arff_scores_SASI = args.arff_scores_SASI

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
        if num_proc == 0:
            print(f"Analysis time too short ({nb_days_analysis} days).")
            print(f"Minimum analysis time: 181 days.")
        return 

    date_start_str = date_start.strftime("%d/%m/%Y")
    date_end_str = date_end.strftime("%d/%m/%Y")

    # M between 10 and 20
    M = args.M
    if M < 10:
        M = 10
    elif M > 20:
        M = 20

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

    # maxiter between 100 and 500
    maxiter = args.maxiter
    if maxiter < 100:
        maxiter = 100
    elif maxiter > 500:
        maxiter = 500

    # Other parameters
    minfloat = 1e-7
    r = 1000
    tolerance = 1e-12
    epsilon = 1.0
    nb_symbols = 2*M + 1
    nb_states = 7

    # Creating a folder for each processor
    dir_tmp = f"./proc_{num_proc}"
    os.makedirs(dir_tmp, exist_ok=True)

    # Temporary file of each processor
    file_hmm = f"{dir_tmp}/hmm_customers_{num_proc}.dat"

    # Reading of the customer data by the master
    if num_proc == 0:
        # ======
        # MASTER
        # ======

        # Start of timing of the time cost
        start = time.perf_counter()
        start_step = time.perf_counter()

        print(f"\nAnalysis from {date_start_str} to {date_end_str}")
		
        customers = read_customer_data(dataset, date_start_str, date_end_str, file_mc, valid_dataset, M, T_min, T_max, min_amount)
        print(f"\nNumber of valid customers :{len(customers)}\n")
        end_step = time.perf_counter()
        print(f"Data access time: {end_step - start_step:g} s")
        start_step = time.perf_counter()

        # customer partitioning
        blocs = partition_customers(customers, nb_procs)

        # sending blocks of customers to slaves
        for p in range(1, nb_procs):
            comm.send(blocs[p], dest=p)

        customers_local = blocs[0]
		
    else:
        # ======
        # SLAVES
        # ======

        # reception of the block of customers sent by the master
        customers_local = comm.recv(source=0)


    # ==============
    # ALL PROCESSORS
    # ==============

    # Creating the customer local HMM file
    f = open(file_hmm, "w", encoding="utf-8")
    f.close()
	
    # Construction of the initial HMMs of the local customers
    initial_hmms = initial_hmms_customers(num_proc, customers_local, nb_states, nb_symbols, epsilon, minfloat)

    # Local customer HMM training
    features_local = train_hmms_customers(customers_local, initial_hmms, maxiter, minfloat, r, tolerance, file_hmm)

    # Agregation of local results
    local_results = (customers_local, features_local)

    # Transmission of local results to the master
    if num_proc != 0:
        # ======
        # SLAVES
        # ======
		
        comm.send(local_results, dest=0)
		
    else:
        # ======
        # MASTER
        # ======
		
        customers_total = []
        features_total = []

        # inserting the master's results
        customers_total += customers_local
        features_total += features_local

        # reception of local results transmitted by slaves
        for p in range(1, nb_procs):

            local_results_p = comm.recv(source=p)

            customers_p, features_p = local_results_p

            customers_total += customers_p
            features_total += features_p
			
        end_step = time.perf_counter()
        print(f"\nHMM-based computation time: {end_step - start_step:g} s")
        start_step = time.perf_counter()

        # Calculation of quadratic coefficients
        coefs_SI, coefs_SA = coefficients_SASI(nb_states, M)

        # Calculating customer scores
        scores_SASI = compute_scores_SASI(customers_total, features_total, coefs_SI, coefs_SA, T_max, csv_scores_SASI)
        end_step = time.perf_counter()
        print(f"\nScore computation time: {end_step - start_step:g} s")

        # Building the ARFF customer file for clustering in Weka
        build_arff(customers_total, scores_SASI, arff_scores_SASI)

        # Saving the start and end dates of the analysis in a file
        with open("./dates.txt", "w") as f:
            f.write(date_start_str + "\n")
            f.write(date_end_str + "\n")

        # End of the timing of the time cost
        end = time.perf_counter()
        print(f"\nTime complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	