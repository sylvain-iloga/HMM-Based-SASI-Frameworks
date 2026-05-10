import pandas as pd
import arff
import argparse


def arff_to_dataframe(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = arff.load(f)

    columns = [attr[0] for attr in dataset["attributes"]]

    df = pd.DataFrame(dataset["data"], columns=columns)

    return df


def copy_column(original_file, clustered_file, csv_file):

    # Loading arff files
    df1 = arff_to_dataframe(original_file)
    df2 = arff_to_dataframe(clustered_file)

    # Copying the cluster column
    df1["cluster"] = df2["cluster"]
    df1["cluster"] = "c" + (df1["cluster"].str.replace("cluster", "").astype(int)).astype(str)

    # Saving the result in a CSV file
    df1["nb_transactions"] = df1["nb_transactions"].astype(int)
    df1.to_csv(csv_file, index=False)

# ============
# Main program
# ============
def main():  
    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="Copy one column from an ARFF file to another")

    parser.add_argument("--original_file", type=str, default="./original_file.arff", help="Original ARFF file")
    parser.add_argument("--clustered_file", type=str, default="./clustered_file.arff", help="Clustered ARFF file")
    parser.add_argument("--csv_file", type=str, default="./csv_file.csv", help="CSV file")

    args = parser.parse_args()

    # Files taken as input parameters
    original_file = args.original_file
    clustered_file = args.clustered_file
    csv_file = args.csv_file
	
    # Copy of the column 'cluster'
    copy_column(original_file, clustered_file, csv_file)

if __name__ == "__main__":
    main()
	