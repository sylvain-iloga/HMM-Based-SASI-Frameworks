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
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse


# =================================================
# 1. Reading the clustering results from a CSV file
# =================================================
def read_clustering_results(file_clustering):

    clusters = []
    matricules = []
    scores_RFM = []
    nb_transactions = []
	
    with open(file_clustering, mode='r', newline='') as f:

        reader = csv.reader(f)

        for line in reader:
            if line[0] == "matricule":
                continue
				
            matricule = line[0]
            score_R = float(line[1])
            score_F = float(line[2])
            score_M = float(line[3])
            cluster = int(line[4].removeprefix("c"))-1

            matricules.append(matricule)
            scores_RFM.append((score_R, score_F, score_M))
            nb_transactions.append(int(score_F))
            clusters.append(cluster)

    labels = np.array(clusters)
    clusters_uniques, cluster_sizes = np.unique(labels, return_counts=True)
    nb_clusters = len(clusters_uniques)

    # Silhouette score
    X = np.array(scores_RFM, dtype=float)
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

    print(f"silhouette: {silhouette:g}\ndb_index: {db_index:g}\nch_index: {ch_index:g}\n")

    return matricules, scores_RFM, nb_transactions, clusters, nb_clusters, cluster_sizes, silhouette, db_index, ch_index
	
# =================================================
# 2. Static PDF visualization of the dataset in R^3
# =================================================
def visualize_customers_pdf(nb_clusters, scores_RFM, clusters, cluster_sizes, date_start_str, date_end_str, silhouette, db_index, ch_index, file_pdf):

    # Extracting coordinates R, F and M
    X = np.array(scores_RFM, dtype=float)
    R = X[:, 0]
    F = X[:, 1]
    M = X[:, 2]

    # Exact limits
    Rmin, Rmax = np.min(R) - 0.05, np.max(R) + 0.05
    Fmin, Fmax = np.min(F) - 0.05, np.max(F) + 0.05
    Mmin, Mmax = np.min(M) - 0.05, np.max(M) + 0.05

    # name of each cluster
    cluster_name = [f"c{k+1}" for k in range(nb_clusters)]

    # Creation of the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
	
    # Different markers per cluster
    markers = ["o","s","p","d","*","+","<",">","v"]

    # Displaying the scatter plot
    clusters = np.array(clusters)
    for k in np.unique(clusters):
        mask = clusters == k
        legend = f"{cluster_name[k]} -> {cluster_sizes[k]}"
		
        ax.scatter(
            R[mask],
            F[mask],
            M[mask],
            marker=markers[k % len(markers)],
            s=10,
            c=np.full(np.sum(mask), k),
            cmap="tab10",
            vmin=np.min(clusters),
            vmax=np.max(clusters),
            alpha=0.8,
            label=legend
        )

    # Fixing the limits
    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Fmin, Fmax)
    ax.set_zlim(Mmin, Mmax)

    # Legend
    ax.set_xlabel("Recency (R)")
    ax.set_ylabel("Frequency (F)")
    ax.set_zlabel("Monetary (M)")
	
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,f"Analysis from {date_start_str} to {date_end_str}", ha="center", fontsize=10)
    fig.text(0.5, 0.02,f"silhouette: {silhouette:g},  dbi: {db_index:g},  chi: {ch_index:g}", ha="center", fontsize=10)
    
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3
    )

    # Saving the image in the PDF format
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(file_pdf, format="pdf")

    # Closing the figure
    plt.close(fig)	

# ====================================================
# 3. Dynamic HTML visualization of the dataset in R^3
# ====================================================
def visualize_customers_html(nb_clusters, scores_RFM, nb_transactions, clusters, cluster_sizes, matricules, date_start_str, date_end_str, silhouette, db_index, ch_index, file_html):

    # Extracting coordinates R, F and M
    X = np.array(scores_RFM, dtype=float)
    R = X[:, 0]
    F = X[:, 1]
    M = X[:, 2]

    # name of each cluster
    cluster_name = [f"c{k+1}" for k in range(nb_clusters)]

    # List of customer matricules
    matricules = np.array(matricules)

    # List of customer statuses
    statuts = np.where(M == 0, "neutral", np.where(M > 0, "contributor", "consumer"))

    # List of customer transaction numbers
    nb_transactions = np.array(nb_transactions)
	
    # Different markers per cluster
    markers = [
        "circle", "square", "diamond",
        "cross", "x", "circle-open",
        "square-open", "diamond-open"
    ]

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22"
    ]
	
    # Creation of the figure
    fig = go.Figure()

    # Displaying the scatter plot
    clusters = np.array(clusters)
    clusters_uniques = np.unique(clusters)
    for k in clusters_uniques:
        k = int(k)
        mask = clusters == k
        legend = f"{cluster_name[k]} -> {cluster_sizes[k]}"
		
        # Data to include in the tooltip
        customdata = np.column_stack((
            matricules[mask],
            statuts[mask],
            nb_transactions[mask]
        ))

        fig.add_trace(go.Scatter3d(
            x=R[mask],
            y=F[mask],
            z=M[mask],
            mode="markers",
            name=legend,
            marker=dict(
                size=4,
                symbol=markers[k % len(markers)],
                color=colors[k % len(colors)]
            ),
            customdata=customdata,
            # Customer tooltip
            hovertemplate=(
                "<b>Matricule:</b> %{customdata[0]}<br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<b>Transactions:</b> %{customdata[2]}<br>"
                "<b>R:</b> %{x:.0f}<br>"
                "<b>F:</b> %{y:.0f}<br>"
                "<b>M:</b> %{z:.0f}"
                "<extra></extra>"
            )
        ))

    # Legend
    fig.update_layout(
        scene=dict(
            xaxis_title="Recency (R)",
            yaxis_title="Frequency (F)",
            zaxis_title="Monetary (M)"
        ),
        plot_bgcolor="#eff6ff",
        width=900,
        height=900,
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(t=120, b=120)
    )

    # Title below the figure
    fig.add_annotation(
        text= f"Analysis from {date_start_str} to {date_end_str}",
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12)
    )

    # Mertics below the title
    fig.add_annotation(
        text= f"silhouette: {silhouette:g},  dbi: {db_index:g},  chi: {ch_index:g}",
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12)
    )

    # Saving the image in the HTML format
    fig.write_html(file_html, include_plotlyjs="cdn")


# ============
# Main program
# ============
def main():

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="3D visualization of static RFM-based customer segmentation")

    parser.add_argument("--csv_clustering", type=str, default="./clustering_customers.csv", help="CSV file containing the results of customer clustering")
    parser.add_argument("--pdf_clustering", type=str, default="./clustering_customers.pdf", help="Static PDF file containing the results of customer clustering")
    parser.add_argument("--html_clustering", type=str, default="./clustering_customers.html", help="Dynamic HTML file containing the results of customer clustering")

    args = parser.parse_args()

    # Files taken as input parameters
    csv_clustering = args.csv_clustering
    pdf_clustering = args.pdf_clustering
    html_clustering = args.html_clustering


    # Reading the start and end dates of the analysis from a file
    with open("./dates.txt", "r") as f:
        date_start_str = f.readline().strip()
        date_end_str = f.readline().strip()

    # Start of timing of the time cost
    start = time.perf_counter()

    # Reading the clustering
    matricules, scores_RFM, nb_transactions, clusters, nb_clusters, cluster_sizes, silhouette, db_index, ch_index = read_clustering_results(csv_clustering)
     
    # Static PDF visualization of the dataset in R^3
    visualize_customers_pdf(nb_clusters, scores_RFM, clusters, cluster_sizes, date_start_str, date_end_str, silhouette, db_index, ch_index, pdf_clustering)

    # Dynamic HTML visualization of the dataset in R^3
    visualize_customers_html(nb_clusters, scores_RFM, nb_transactions, clusters, cluster_sizes, matricules, date_start_str, date_end_str, silhouette, db_index, ch_index, html_clustering)

    # Saving the metrics in a file
    with open("./metrics.txt", "w") as f:
        f.write(f"silhouette: {silhouette:g}\n")
        f.write(f"db_index: {db_index:g}\n")
        f.write(f"ch_index: {ch_index:g}\n")

    # End of the timing of the time cost
    end = time.perf_counter()
    print(f"Time complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	