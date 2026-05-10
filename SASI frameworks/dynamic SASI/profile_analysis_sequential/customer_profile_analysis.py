import sys
import re
import os
import csv
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import json
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

    if nb_days <= 7:
        # between 0 day and 1 week
        x = 0
    elif 8 <= nb_days <= 60:
        # between 1 week and 2 months
        x = 1
    elif 61 <= nb_days <= 180:
        # between 2 months and 6 months
        x = 2
    else:
        # beyond 6 months
        x = 3

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

# ==================================================
# 7. Decomposition of the Markov chains of customers
# ==================================================
def decompose_mc(long_mc, size, overlap):

    # Extraction of the pairs
    pairs = re.findall(r'\(\d+,\d+\)', long_mc)

    list_mcs = []
    step = size - overlap  

    for i in range(0, len(pairs), step):
        # Extraction of the current MC
        mc = pairs[i:i + size]
        if not mc:
            break

        # The last MC must have the correct size 
        if len(mc) != size:
            mc = pairs[i-size+len(mc):i+len(mc)]

        # Saving the current MC
        list_mcs.append(''.join(mc))

        if i + size >= len(pairs):
            break

    return list_mcs

# ====================================================
# 8. Construction of the initial HMMs of the customers
# ====================================================
def initial_hmms_customers(customers, N, M, size, overlap, epsilon, minfloat):

    initial_hmms = []
	
    for customer in customers:
        long_mc = ''.join(f'({x},{y})' for (x,y) in customer.markov_chain) 

        # Decomposing the long MC for the sliding window mechanism
        list_mcs = decompose_mc(long_mc, size, overlap)

        # initializing the HMM of each sub-window
        initial_hmms_customer = []
        for mc in list_mcs:
            mc = mc + "\n"
            with open("tmp.txt", "w", encoding="utf-8") as g:
                g.write(mc)  
				
            A, B, pi, sequences = initial_hmm("tmp.txt", N, M, epsilon, minfloat)
            initial_hmms_customer.append((A, B, pi, sequences))
			
        initial_hmms.append(initial_hmms_customer)

    os.remove("tmp.txt")

    return initial_hmms

# ====================================================
# 9. Constructing a HMM object with pomegranate 0.14.8
# ====================================================
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

# ================================================================================
# 10. Extracting parameters (A, B, pi) from an HMM object using pomegranate 0.14.8
# ================================================================================
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
# 11. Training an HMM object with pomegranate 0.14.8
# ==================================================
def train_hmm_pomegranate(hmm_init, sequences, maxiter, minfloat):
    # 1. Baum-Welch algorithm
    hmm_final = hmm_init.fit(sequences, algorithm='baum-welch', max_iterations=maxiter, verbose=False)
    hmm_final.bake()

    # 2. Extracting the final parameters
    A_bar, B_bar, pi_bar = parameters_hmm_pomegranate(hmm_final, minfloat)

    return A_bar, B_bar, pi_bar
		
# ================================
# 12. Vector associated with a HMM
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
# 13. Saving a customer's HMM in a text file
# ==========================================
def save_hmm(matricule, index_hmm, A, B, pi, phi, phi_bar, file_hmm):
    separator = "========================================================"
    separator_customers = "########################################################"

    with open(file_hmm, "a") as f:

        # 1. Matricule
        f.write(separator_customers + "\n")
        f.write(matricule + " (" + str(index_hmm) +  ")\n")
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
# 14. Customers' HMM training
# ===========================
def train_hmms_customers(customers, initial_hmms, maxiter, minfloat, r, tolerance, file_hmm, csv_omega_bar, csv_omega):
    
    features = []

    for initial_hmms_customer, customer in zip(initial_hmms, customers):  
        distances_phi = []
        distances_phi_bar = []
        for index_hmm, (A_init, B_init, pi_init, sequences) in enumerate(initial_hmms_customer):  
            hmm_init = build_hmm_pomegranate(A_init, B_init, pi_init)
            A, B, pi = train_hmm_pomegranate(hmm_init, sequences, maxiter, minfloat)
            phi, phi_bar = vector_hmm(A, B, minfloat, r, tolerance)
            if index_hmm != 0:
                # Normalized Euclidean distance between the old and the current phi
                distance_phi = np.linalg.norm(phi - old_phi)/np.sqrt(2)
                distances_phi.append(distance_phi)
                # Normalized Euclidean distance between the old and the current phi_bar
                distance_phi_bar = np.linalg.norm(phi_bar - old_phi_bar)/np.sqrt(2)
                distances_phi_bar.append(distance_phi_bar)

            save_hmm(customer.matricule, index_hmm+1, A, B, pi, phi, phi_bar, file_hmm)
            old_phi = np.copy(phi)
            old_phi_bar = np.copy(phi_bar)

        omega = np.array(distances_phi, dtype=float)
        omega_bar = np.array(distances_phi_bar, dtype=float)
        features.append((omega, omega_bar))

        # Saving the curves of omega and omega_bar in files
        with open(csv_omega_bar, "a", encoding="utf-8") as f, \
             open(csv_omega, "a", encoding="utf-8") as g:
		
            f.write(customer.matricule + ",")
            values = ",".join("%g" % dist for dist in omega_bar)
            f.write(values + "\n")
		
            g.write(customer.matricule + ",")
            values = ",".join("%g" % dist for dist in omega)
            g.write(values + "\n")

    return features

# ===================================
# 15. Instability level of a customer
# ===================================
def instability_level(mu):

    if 0.0 <= mu <= 0.05:
        mu_bar = 0
    elif 0.05 < mu <= 0.10:
        mu_bar = 1
    elif 0.10 < mu <= 0.15:
        mu_bar = 2
    elif 0.15 < mu <= 0.20:
        mu_bar = 3
    elif 0.20 < mu <= 0.25:
        mu_bar = 4
    elif 0.25 < mu <= 0.35:
        mu_bar = 5
    elif 0.35 < mu <= 0.45:
        mu_bar = 6
    elif 0.45 < mu <= 0.60:
        mu_bar = 7
    elif 0.60 < mu <= 0.80:
        mu_bar = 8
    else:
        mu_bar = 9

    return mu_bar 

# ==========================================
# 16. Partial instability rate of a customer 
# ==========================================
def partial_instability_rate(omega_bar, omega):

    surface_under_omega_bar = np.sum((omega_bar[:-1] + omega_bar[1:]) / 2.0)
    surface_norm_under_omega_bar = surface_under_omega_bar / (len(omega_bar)-1)

    surface_under_omega = np.sum((omega[:-1] + omega[1:]) / 2.0)
    surface_norm_under_omega = surface_under_omega / (len(omega)-1)

    gamma_bar = surface_norm_under_omega_bar
    gamma = surface_norm_under_omega

    return gamma_bar, gamma	

# ===========================================
# 17.Instability rate and level of a customer 
# ===========================================
def instability_rate_and_level(omega_bar, omega, recent=0.65, amounts=0.65):

    omega_bar = np.asarray(omega_bar, dtype=float)
    omega = np.asarray(omega, dtype=float)

    # Analysis of the two halves of omega_bar and omega
    T = len(omega_bar) // 2
    # All except the last (T+1) elements (old behavior)
    omega_bar_old = omega_bar[:-(T+1)]	
    omega_old = omega[:-(T+1)]	
    # Only the last (T+1) elements (recent behavior)
    omega_bar_recent = omega_bar[-(T+1):]	
    omega_recent = omega[-(T+1):]

    # Calculation of the instability rates for old and recent behaviors
    gamma_bar_old, gamma_old = partial_instability_rate(omega_bar_old, omega_old)
    gamma_bar_recent, gamma_recent = partial_instability_rate(omega_bar_recent, omega_recent)


    # final instability rate and level
    gamma_bar = recent * gamma_bar_recent + (1.0 - recent) * gamma_bar_old
    gamma = recent * gamma_recent + (1.0 - recent) * gamma_old
    mu = amounts * gamma_bar + (1.0 - amounts) * gamma
    mu_bar = instability_level(mu)

    return gamma_bar, gamma, mu_bar, mu	

# ====================================================
# 18.Instability rates and levels of all the customers 
# ====================================================
def instability_rate_and_level_customers(customers, features, file_instability):

    instability_rates = []
    instability_rates_SASI = []
    instability_levels = []

    for (omega, omega_bar) in features:  

        gamma_bar, gamma, mu_bar, mu = instability_rate_and_level(omega_bar, omega)

        instability_rates_SASI.append((gamma_bar, gamma))
        instability_rates.append(mu)
        instability_levels.append(mu_bar)
	
    with open(file_instability, "w") as f:
        f.write("matricule,mu_bar,mu,gamma_bar,gamma\n")
        for customer, (gamma_bar, gamma), mu, mu_bar  in zip(customers, instability_rates_SASI, instability_rates, instability_levels):  
            f.write(customer.matricule + ",")
            f.write(f"{mu_bar},")
            f.write(f"{mu:g},")
            f.write(f"{gamma_bar:g},")
            f.write(f"{gamma:g}\n")

    return instability_rates_SASI, instability_levels, instability_rates

# =========================================================
# 19. Static PDF visualization of customer profile in R^2
# =========================================================
def visualize_customers_profile_pdf(instability_rates_SASI, instability_rates, customers, features, instability_levels, date_start_str, date_end_str, window_size, overlap, pdf_files):

    # sorting customers in descending order of instability scores
    sorted_data = sorted(zip(customers, features, instability_rates_SASI, instability_levels, instability_rates),
                     key=lambda x: x[4],
                     reverse=True)

    # Editing the PDF page by page, one page per customer
    with PdfPages(pdf_files) as pdf:

        for customer, (omega, omega_bar), (gamma_bar, gamma), mu_bar, mu in sorted_data:

            omega_norm = 100 * omega
            omega_bar_norm = 100 * omega_bar

            time_max = len(omega_norm)+1
            t = np.arange(1, time_max)

            fig, ax = plt.subplots(figsize=(8, 5))

            # Fixing the limits
            ax.set_xlim(0, time_max)
            ax.set_ylim(0, 100)

            # Legend
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Instability (in %)")
	
            # Curve omega_bar(t): blue, thick solid line
            ax.plot(
                t, omega_bar_norm,
                color="blue",
                linewidth=2.5,
                linestyle="-",
                label="SA" 
            )

            # Curve omega(t): red, thick dashed line
            ax.plot(
                t, omega_norm,
                color="red",
                linewidth=2.5,
                linestyle="--",
                label="SI"
            )

            # Grid
            ax.grid(True)

            # Legend above
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=2
            )

            # Reserve space below
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            # customer's matricule  and instability level
            msg = f"[Matricule: {customer.matricule}]   [Instability: {mu_bar}]"

            fig.text(
                0.5, 0.03,   
                msg,
                ha="center",
                fontsize=12
            )

            # Analysis period, number of transactions and window size
            msg = f"From {date_start_str} to {date_end_str}\nTransactions: {customer.nb_transactions}\nWindow size: {window_size} - {overlap}"

            fig.text(
                0.97, 0.025,
                msg,
                ha="right",
                fontsize=8,
                color="gray"
            )
            
            # Instability rates gamma_bar, gamma and mu
            msg = f"Instability_SA: {gamma_bar*100.0:.2f}%\nInstability_SI: {gamma*100.0:.2f}%\nInstability: {mu*100.0:.2f}%"

            fig.text(
                0.05, 0.025,
                msg,
                ha="left",
                fontsize=8,
                color="gray"
            )

            pdf.savefig(fig)
            plt.close(fig)


# ===========================================================
# 20. Dynamic HTML visualization of customer profile in R^2
# ===========================================================
def visualize_customers_profile_html(instability_rates_SASI, instability_rates, customers, features, instability_levels, date_start_str, date_end_str, window_size, overlap, file_html):

    # sorting customers in descending order of instability scores
    sorted_data = sorted(
        zip(customers, features, instability_rates_SASI, instability_levels, instability_rates),
        key=lambda x: x[4],
        reverse=True
    )

    data_customers = []

    # Editing the HTML page displaying the data of only one customer at a time
    for customer, (omega, omega_bar), (gamma_bar, gamma), mu_bar, mu in sorted_data:

        omega_norm = (100 * omega).tolist()
        omega_bar_norm = (100 * omega_bar).tolist()

        t = list(range(1, len(omega_norm) + 1))

        data_customers.append({
            "matricule": str(customer.matricule),
            "nb_transactions": int(customer.nb_transactions),
            "mu_bar": int(mu_bar),
            "gamma_bar": float(gamma_bar*100),
            "gamma": float(gamma*100),
            "mu": float(mu*100),
            "t": list(map(int, t)),
            "omega": list(map(float, omega_norm)),
            "omega_bar": list(map(float, omega_bar_norm))
        })
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body {{ font-family: Arial; }}

#container {{
    width: 900px;
    margin: auto;
    position: relative;
}}

#plot {{
    width: 100%;
    height: 500px;
}}

.nav-btn {{
    position: absolute;
    bottom: 10px;
    font-size: 28px;
    background: lightgray;
    border: none;
    cursor: pointer;
    padding: 5px 10px;
    z-index: 10;
}}

.nav-btn:disabled {{
    opacity: 0.3;
    cursor: not-allowed;
}}

/* Wide offset to avoid collision */
#first {{ left: -40px; }}
#prev  {{ left: 20px; }}

#next  {{ right: 20px; }}
#last  {{ right: -40px; }}

/* Counter */
#counter {{
    position: absolute;
    top: 63px;
    right: 80px;
    font-size: 14px;
    z-index: 1000;
    background: white;
    padding: 2px 6px;
    border-radius: 4px;
}}

#search-box {{
    position: absolute;
    top: 63px;
    left: 80px;
    z-index: 1000;
    background: white;
    padding: 2px 6px;
    border-radius: 4px;
}}

#search-input {{
    padding: 3px;
    font-size: 12px;
    width: 140px;
}}

#search-btn {{
    font-size: 14px;
    cursor: pointer;
}}

</style>
</head>

<body>

<div id="container">
    <div id="counter"></div>
    <div id="plot"></div>
	
<div id="search-box">
    <input type="text" id="search-input" placeholder="Matricule or #index">
    <button id="search-btn">🔍</button>
</div>

<button id="first" class="nav-btn">⏮</button>
<button id="prev"  class="nav-btn">◀</button>
<button id="next"  class="nav-btn">▶</button>
<button id="last"  class="nav-btn">⏭</button>
</div>

<script>

let data_customers = {json.dumps(data_customers, ensure_ascii=False)};

let matricule_to_index = {{}};
for (let i = 0; i < data_customers.length; i++) {{
    matricule_to_index[data_customers[i].matricule] = i;
}}

let date_start = {json.dumps(date_start_str)};
let date_end = {json.dumps(date_end_str)};
let index = 0;

function update_buttons() {{
    document.getElementById("prev").disabled  = (index === 0);
    document.getElementById("first").disabled = (index === 0);

    document.getElementById("next").disabled  = (index === data_customers.length - 1);
    document.getElementById("last").disabled  = (index === data_customers.length - 1);
}}

function update_counter() {{
    const el = document.getElementById("counter");
    el.innerHTML = "<b>Customer " + (index + 1) + "/" + data_customers.length + "</b>";
}}

function display_customer(i) {{

    let c = data_customers[i];

    let trace_SA = {{
        x: c.t,
        y: c.omega_bar,
        mode: 'lines',
        name: 'SA',
        line: {{color: 'blue', width: 3}}
    }};

    let trace_SI = {{
        x: c.t,
        y: c.omega,
        mode: 'lines',
        name: 'SI',
        line: {{color: 'red', width: 3, dash: 'dash'}}
    }};

    let layout = {{
        margin: {{ t: 105 }}, 
        xaxis: {{
            title: "Time (t)",
            range: [0, Math.max(...c.t) + 1]
        }},
        yaxis: {{
            title: "Instability (in %)",
            range: [0, 100]
        }},
        plot_bgcolor: "#eff6ff",
        legend: {{
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: 1.15
        }},
        annotations: [

            {{
                text: "[Matricule: <b>" + c.matricule + "</b>]   [Instability: <b>" + c.mu_bar + "</b>]",
                x: 0.5,
                y: -0.25,
                xref: "paper",
                yref: "paper",
                showarrow: false
            }},

            {{
                text:
                "From " + date_start + " to " + date_end + "<br>" +
                "Transactions: " + c.nb_transactions + "<br>" +
                "Windows size: {window_size} - {overlap}",
                x: 1,
                y: -0.25,
                xref: "paper",
                yref: "paper",
                showarrow: false,
                xanchor: "right",
                font: {{size: 10, color: "gray"}}
            }},

            {{
                text:
                "Instability_SA: " + c.gamma_bar.toFixed(2) + "%<br>" +
                "Instability_SI: " + c.gamma.toFixed(2) + "%<br>" +
                "Instability: " + c.mu.toFixed(2) + "%",
                x: 0,
                y: -0.25,
                xref: "paper",
                yref: "paper",
                showarrow: false,
                xanchor: "left",
                font: {{size: 10, color: "gray"}}
            }}
        ]
    }};

    Plotly.newPlot('plot', [trace_SA, trace_SI], layout, {{displayModeBar: false}});

    update_buttons();
    update_counter();
}}


function search_customer() {{

    let input = document.getElementById("search-input").value.trim();

    if (input.length === 0) return;

    // CASE 1 : search by index (#x)
    if (input.startsWith("#")) {{

        let num = parseInt(input.substring(1));

        if (!isNaN(num) && num >= 1 && num <= data_customers.length) {{
            index = num - 1;
            display_customer(index);
        }}

        return;
    }}

    // CASE 2 : search by matricule
    if (input in matricule_to_index) {{
        index = matricule_to_index[input];
        display_customer(index);
    }}

    // CASE 3 : otherwise -> nothing
}}

document.getElementById("prev").onclick = function() {{
    if (index > 0) {{
        index--;
        display_customer(index);
    }}
}};

document.getElementById("next").onclick = function() {{
    if (index < data_customers.length - 1) {{
        index++;
        display_customer(index);
    }}
}};

document.getElementById("first").onclick = function() {{
    index = 0;
    display_customer(index);
}};

document.getElementById("last").onclick = function() {{
    index = data_customers.length - 1;
    display_customer(index);
}};


document.getElementById("search-btn").onclick = search_customer;

document.getElementById("search-input").addEventListener("keypress", function(e) {{
    if (e.key === "Enter") {{
        search_customer();
    }}
}});


// Initialization
display_customer(index);

</script>

</body>
</html>
"""
    with open(file_html, "w", encoding="utf-8") as f:
        f.write(html)

# ==================================================
# 21. Static PDF visualization of the dataset in R^2
# ==================================================
def visualize_customers_pdf(instability_rates_SASI, instability_levels, nb_instability_levels, date_start_str, date_end_str, window_size, overlap, pdf_file):

    # Size of each cluster
    size_clusters = np.zeros(nb_instability_levels, dtype=int)
    for mu_bar in instability_levels:
	    size_clusters[mu_bar] += 1

    # Extracting coordinates X and Y
    dataset = (100 * np.array(instability_rates_SASI, dtype=float))
    X = dataset[:, 0]
    Y = dataset[:, 1]

    # Exact limits
    Xmin, Xmax = np.min(X) - 0.05, np.max(X) + 0.05
    Ymin, Ymax = np.min(Y) - 0.05, np.max(Y) + 0.05

    # Creation of the figure
    fig, ax = plt.subplots(figsize=(8,8))
	
    # Different markers per cluster
    markers = ["o","s","p","d","*","+","<",">","v"]

    # name of each cluster
    cluster_name = [f"inst_{k}" for k in range(nb_instability_levels)]

    # Displaying the scatter plot
    for k in np.unique(instability_levels):
        mask = instability_levels == k
        legend = f"{cluster_name[k]} -> {size_clusters[k]}"
		
        ax.scatter(
            X[mask],
            Y[mask],
            marker=markers[k % len(markers)],
            s=10,
            c=np.full(np.sum(mask), k),
            cmap="tab10",
            vmin=np.min(instability_levels),
            vmax=np.max(instability_levels),
            alpha=0.8,
            label=legend
        )

    # Fixing the limits
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)

    # Legend
    ax.set_xlabel("Instability SA (in %)")
    ax.set_ylabel("Instability SI (in %)")
	
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.02,f"Analysis from {date_start_str} to {date_end_str}\nWindows size: {window_size} - {overlap}", ha="center", fontsize=12)
    
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3
    )

    # Saving the image in the PDF format
    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.94])
    fig.savefig(pdf_file, format="pdf")

    # Closing the figure
    plt.close(fig)	

# ====================================================
# 22. Dynamic HTML visualization of the dataset in R^2
# ====================================================
def visualize_customers_html(instability_rates_SASI, instability_rates, instability_levels, nb_instability_levels, customers, date_start_str, date_end_str, window_size, overlap, file_html):

    # Size of each cluster
    size_clusters = np.zeros(nb_instability_levels, dtype=int)
    for mu_bar in instability_levels:
        size_clusters[mu_bar] += 1

    # Extracting coordinates X and Y
    dataset = (100 * np.array(instability_rates_SASI, dtype=float))
    X = dataset[:, 0].tolist()
    Y = dataset[:, 1].tolist()

    # Exact limits
    Xmin, Xmax = np.min(X) - 0.05, np.max(X) + 0.05
    Ymin, Ymax = np.min(Y) - 0.05, np.max(Y) + 0.05

    # List of matricules
    matricules = [customer.matricule for customer in customers]
    matricules = list(map(str, matricules))

    # Number of transactions
    nb_transactions = [customer.nb_transactions for customer in customers]

    # Instability rates
    instability_rates = (100 * np.array(instability_rates, dtype=float)).tolist()

    # Cluster names
    cluster_name = [f"level {k}" for k in range(nb_instability_levels)]

    # Preparing data for JavaScript
    data_points = []
    for i in range(len(matricules)):
        data_points.append({
            "matricule": matricules[i],
            "X": X[i],
            "Y": Y[i],
            "cluster": int(instability_levels[i]),
            "cluster_name": cluster_name[int(instability_levels[i])],
            "transactions": nb_transactions[i],
            "instability_rate": instability_rates[i]
        })

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body {{ font-family: Arial; }}

#container {{
    width: 850px;
    margin: auto;
}}

#plot {{
    width: 100%;
    height: 850px;
}}

#search-bar {{
    display: flex;
    justify-content: flex-start;
    align-items: center;
    gap: 30px;
    margin-bottom: 5px;
    margin-top: 25px;
}}

#result {{
    font-weight: bold;
    color: black;
}}
</style>
</head>

<body>

<div id="container">

    <div id="search-bar">
        <div>
            <input type="text" id="searchBox" placeholder="Matricule" style="padding:5px;">
            <button onclick="searchClient()">🔍</button>
        </div>
        <div id="result"></div>
    </div>

    <div id="plot"></div>

</div>

<script>

let data = {json.dumps(data_points)};

// =======================
// Build clusters
// =======================
let clusters = {{}};
data.forEach(d => {{
    if (!(d.cluster in clusters)) {{
        clusters[d.cluster] = [];
    }}
    clusters[d.cluster].push(d);
}});

// =======================
// Create traces + index map
// =======================
let traces = [];
let pointIndexMap = {{}}; // matricule -> (traceIndex, pointIndex)

let traceIndex = 0;

let markers = [
    "circle", "square", "pentagon",
    "diamond", "star", "cross",
    "triangle-left", "triangle-right", "triangle-down"
];

let colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22"
];

Object.keys(clusters).forEach(k => {{
    let pts = clusters[k];

    pts.forEach((p, i) => {{
        pointIndexMap[p.matricule] = {{
            traceIndex: traceIndex,
            pointIndex: i
        }};
    }});

    traces.push({{
        x: pts.map(p => p.X),
        y: pts.map(p => p.Y),
        mode: 'markers',
        name: pts[0].cluster_name + " -> " + pts.length,
        text: pts.map(p =>
            "Matricule: " + p.matricule +
            "<br>Transactions: " + p.transactions +
            "<br>Instability_SA: " + p.X.toFixed(2) +
            "<br>Instability_SI: " + p.Y.toFixed(2) +
            "<br>Instability: " + p.instability_rate.toFixed(2)
        ),
        hoverinfo: "text",
        marker: {{
            size: 6,
            symbol: markers[parseInt(k) % markers.length],
            color: colors[parseInt(k) % colors.length]
        }}
    }});

    traceIndex++;
}});

// =======================
// Layout
// =======================
let layout = {{
    xaxis: {{title: "Instability SA (in %)", range: [{Xmin}, {Xmax}]}},
    yaxis: {{title: "Instability SI (in %)", range: [{Ymin}, {Ymax}]}},
    legend: {{
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: 1.1
    }},
    plot_bgcolor: "#eff6ff",
    hovermode: "closest",
    margin: {{t: 100, b: 120}}
}};

Plotly.newPlot("plot", traces, layout);

// =======================
// Search + focus + hover
// =======================
function searchClient() {{
    let m = document.getElementById("searchBox").value.trim();
    let res = document.getElementById("result");

    let found = data.find(d => d.matricule === m);

    if (found) {{
        res.innerHTML = found.cluster_name + " (" + found.X.toFixed(2) + " , " + found.Y.toFixed(2) + ")";

        let plot = document.getElementById("plot");

        // forced hover 
        let idx = pointIndexMap[m];
        Plotly.Fx.hover(plot, [{{
            curveNumber: idx.traceIndex,
            pointNumber: idx.pointIndex
        }}]);

    }} else {{
        res.innerHTML = "";
    }}
}}

// ENTER key
document.getElementById("searchBox").addEventListener("keypress", function(e) {{
    if (e.key === "Enter") searchClient();
}});

// =======================
// Title below the figure
// =======================
let plotDiv = document.getElementById("plot");
Plotly.addAnnotation(plotDiv, {{
    text: "Analysis from {date_start_str} to {date_end_str}<br>Window size: {window_size} - {overlap}",
    x: 0.5,
    y: -0.18,
    xref: "paper",
    yref: "paper",
    showarrow: false,
    font: {{size: 16}}
}});

</script>

</body>
</html>
"""

    with open(file_html, "w", encoding="utf-8") as f:
        f.write(html)


# ============
# Main program
# ============
def main():

    # Reading the parameters to be passed to command line
    parser = argparse.ArgumentParser(description="Customer profile analysis using HMMs")

    parser.add_argument("--dataset", type=str, default="./customers.txt", help="Customer bank transaction file")
    parser.add_argument("--valid_dataset", type=str, default="./valid_customers.txt", help="Valid customer transaction file")
    parser.add_argument("--file_mc", type=str, default="./mc_customers.dat", help="File containing the Markov chains of the customers")
    parser.add_argument("--file_hmm", type=str, default="./hmm_customers.dat", help="File containing the HMMs of the customers")
    parser.add_argument("--csv_omega_bar", type=str, default="./omega_bar.csv", help="File containing the curves of omega_bar associated with the customers")
    parser.add_argument("--csv_omega", type=str, default="./omega.csv", help="File containing the curves of omega associated with the customers")
    parser.add_argument("--csv_instability_rates_and_levels", type=str, default="./mu_and_mu_bar.csv", help="File containing the instability rates and levels associated with the customers")
    parser.add_argument("--pdf_profiles", type=str, default="./profiles.pdf", help="Static PDF file displaying the profile of each customer")
    parser.add_argument("--html_profiles", type=str, default="./profiles.html", help="Dynamic HTML file displaying the profile of each customer")
    parser.add_argument("--pdf_instability_levels", type=str, default="./instability_levels.pdf", help="Static PDF file containing the instability level of each customer")
    parser.add_argument("--html_instability_levels", type=str, default="./customers.html", help="Dynamic HTML file containing the instability level of each customer")
    parser.add_argument("--date_start_str", type=str, default="01/01/2023", help="Start date of the customer analysis period")
    parser.add_argument("--date_end_str", type=str, default="31/12/2023", help="End date of the customer analysis period")
    parser.add_argument("--M", type=int, default=20, help="The number of symbols is (2*M+1).")
    parser.add_argument("--min_amount", type=int, default=12500, help="Minimum amount for a valid transaction.")
    parser.add_argument("--T_min", type=int, default=300, help="Minimum number of valid transactions.")
    parser.add_argument("--T_max", type=int, default=3000, help="Maximum number of valid recent transactions.")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of iterations for the Baum-Welch algorithm.")

    args = parser.parse_args()

    # Files taken as input parameters
    dataset = args.dataset
    valid_dataset = args.valid_dataset
    file_mc = args.file_mc
    file_hmm = args.file_hmm
    csv_omega_bar = args.csv_omega_bar
    csv_omega = args.csv_omega
    csv_instability_rates_and_levels = args.csv_instability_rates_and_levels
    pdf_profiles = args.pdf_profiles
    html_profiles = args.html_profiles
    pdf_instability_levels = args.pdf_instability_levels
    html_instability_levels = args.html_instability_levels


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

    # T_min between 300 and 3000
    T_min = args.T_min
    if T_min < 300:
        T_min = 300
    elif T_min > 3000:
        T_min = 3000
		
    # T_max between 300 and 3000
    T_max = args.T_max
    if T_max < 300:
        T_max = 300
    elif T_max > 3000:
        T_max = 3000

    # always have (T_min <= T_max) 
    if T_min > T_max:
        T_min, T_max = T_max, T_min

    # maxiter between 100 and 500
    maxiter = args.maxiter
    if maxiter < 100:
        maxiter = 100
    elif maxiter > 500:
        maxiter = 500

    # consecutive sub-windows containing (l = 50) transactions
    window_size = 50

    # with an overlap of (kappa = 5) transactions
    overlap = 5
        
    # Other parameters
    minfloat = 1e-7
    r = 1000
    tolerance = 1e-12
    epsilon = 1.0
    nb_symbols = 2*M + 1
    nb_states = 4
    nb_instability_levels = 10

    # Start of timing of the time cost
    start = time.perf_counter()
    start_step = time.perf_counter()

    # Reading customer data
    customers = read_customer_data(dataset, date_start_str, date_end_str, file_mc, valid_dataset, M, T_min, T_max, min_amount)
    print(f"\nNumber of valid customers :{len(customers)}\n")
    end_step = time.perf_counter()
    print(f"Data access time: {end_step - start_step:g} s")
    start_step = time.perf_counter()
    
    # Creating the customer HMM file
    f = open(file_hmm, "w", encoding="utf-8")
    f.close()
	
    # Creation of the file containing the curves of omega_bar associated with the customers
    f = open(csv_omega_bar, "w", encoding="utf-8")
    f.close()
	
    # Creation of the file containing the curves of omega associated with the customers
    f = open(csv_omega, "w", encoding="utf-8")
    f.close()
	
    # Construction of the initial HMMs of the customers
    initial_hmms = initial_hmms_customers(customers, nb_states, nb_symbols, window_size, overlap, epsilon, minfloat)

    # Customer HMM training
    features = train_hmms_customers(customers, initial_hmms, maxiter, minfloat, r, tolerance, file_hmm, csv_omega_bar, csv_omega)
    end_step = time.perf_counter()
    print(f"\nHMM-based computation time: {end_step - start_step:g} s")
    start_step = time.perf_counter()

    # Calculating customer instability levels
    instability_rates_SASI, instability_levels, instability_rates = instability_rate_and_level_customers(customers, features, csv_instability_rates_and_levels)
    end_step = time.perf_counter()
    print(f"\nInstability computations time: {end_step - start_step:g} s")

    # Static PDF visualization of customer profile in R^2
    visualize_customers_profile_pdf(instability_rates_SASI, instability_rates, customers, features, instability_levels, date_start_str, date_end_str, window_size, overlap, pdf_profiles)

    # Dynamic HTML visualization of customer profile in R^2
    visualize_customers_profile_html(instability_rates_SASI, instability_rates, customers, features, instability_levels, date_start_str, date_end_str, window_size, overlap, html_profiles)

    # Static PDF visualization of customer instability levels in R^2
    visualize_customers_pdf(instability_rates_SASI, instability_levels, nb_instability_levels, date_start_str, date_end_str, window_size, overlap, pdf_instability_levels)

    # Dynamic HTML visualization of customer instability levels in R^2
    visualize_customers_html(instability_rates_SASI, instability_rates, instability_levels, nb_instability_levels, customers, date_start_str, date_end_str, window_size, overlap, html_instability_levels)

    # End of the timing of the time cost
    fin = time.perf_counter()
    print(f"\nTime complexity: {fin - start:g} s")
	
if __name__ == "__main__":
    main()
	