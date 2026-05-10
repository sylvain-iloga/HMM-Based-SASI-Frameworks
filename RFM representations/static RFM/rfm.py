import re
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
import math
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

# =================================
# 2. Save RFM features in ARFF file
# =================================
def save_rfm_arff(customers, date_end_str, file_rfm):

    R_values = []
    F_values = []
    M_values = []
    matricules = []

    date_end = datetime.strptime(date_end_str, "%d/%m/%Y")

    for customer in customers:

        dates = [date_obj for (date_obj, _) in customer.transactions]
        amounts = [amount for (_, amount) in customer.transactions]

        # R = recency 
        # days from the last transaction to the end of the analysis time
        last_date = dates[-1]
        R = (date_end - last_date).days

        # F = frequency
        # number of transactions
        F = customer.nb_transactions

        # M = Monetary
        # Volume
        M = sum([abs(amount) for amount in amounts])

        # Updating the lists
        R_values.append(R)
        F_values.append(F)
        M_values.append(M)
        matricules.append(customer.matricule)

    # Saving RFM features in ARFF file
    with open(file_rfm, "w", encoding="utf-8") as f:

        f.write("@RELATION customers_rfm\n\n")

        f.write("@ATTRIBUTE matricule STRING\n")
        f.write("@ATTRIBUTE R INTEGER\n")
        f.write("@ATTRIBUTE F INTEGER\n")
        f.write("@ATTRIBUTE M REAL\n")

        f.write("@DATA\n")

        for matricule, x, y, z in zip(matricules, R_values, F_values, M_values):
            f.write(f"\"{matricule}\",{int(x)},{int(y)},{z:.1f}\n")

    return R_values, F_values, M_values			
			
# ============
# Main program
# ============
def main():

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="Customer segmentation for a bank using RFM")

    parser.add_argument("--dataset", type=str, default="./customers.txt", help="Customer bank transaction file")
    parser.add_argument("--valid_dataset", type=str, default="./valid_customers.txt", help="Valid customer transaction file")
    parser.add_argument("--file_rfm", type=str, default="./rfm.arff", help="ARFF file containing the RFM scores of the customers")
    parser.add_argument("--date_start_str", type=str, default="01/01/2023", help="Start date of the customer analysis period")
    parser.add_argument("--date_end_str", type=str, default="31/12/2023", help="End date of the customer analysis period")
    parser.add_argument("--min_amount", type=int, default=12500, help="Minimum amount for a valid transaction.")
    parser.add_argument("--T_min", type=int, default=150, help="Minimum number of valid transactions.")
    parser.add_argument("--T_max", type=int, default=500, help="Maximum number of valid recent transactions.")

    args = parser.parse_args()

    # Files taken as input parameters
    dataset = args.dataset
    valid_dataset = args.valid_dataset
    file_rfm = args.file_rfm

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

    # Start of timing of the time cost
    start = time.perf_counter()
    
    # Reading customer data
    customers = read_customer_data(dataset, date_start_str, date_end_str, T_min, T_max, min_amount, valid_dataset)
    print(f"\nNumber of valid customers :{len(customers)}\n")

    # Save RFM coefficients in ARFF file
    save_rfm_arff(customers, date_end_str, file_rfm)

    # Saving the start and end dates of the analysis in a file
    with open("./dates.txt", "w") as f:
        f.write(date_start_str + "\n")
        f.write(date_end_str + "\n")

    # End of the timing of the time cost
    end = time.perf_counter()
    print(f"Time complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	