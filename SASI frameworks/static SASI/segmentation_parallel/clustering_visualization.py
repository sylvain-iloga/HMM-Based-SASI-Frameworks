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
import plotly.utils as pu
import json
import time
import argparse

# =================================================
# 1. Reading the clustering results from a CSV file
# =================================================
def read_clustering_results(file_clustering):

    clusters = []
    matricules = []
    scores_SASI = []
    nb_transactions = []
	
    with open(file_clustering, mode='r', newline='') as f:

        reader = csv.reader(f)

        for line in reader:
            if line[0] == "matricule":
                continue
				
            matricule = line[0]
            x = float(line[1])
            y = float(line[2])
            nb_transacts = int(line[3])
            cluster = int(line[4].removeprefix("c"))-1

            matricules.append(matricule)
            scores_SASI.append((x, y))
            nb_transactions.append(nb_transacts)
            clusters.append(cluster)

    labels = np.array(clusters)
    clusters_uniques, cluster_sizes = np.unique(labels, return_counts=True)
    nb_clusters = len(clusters_uniques)

    # Silhouette score
    dataset = np.array(scores_SASI, dtype=float)
    if nb_clusters > 1 and len(dataset) > nb_clusters:
        silhouette = silhouette_score(dataset, labels)
    else:
        silhouette = None  

    # Davies-Bouldin index
    if nb_clusters > 1:
        db_index = davies_bouldin_score(dataset, labels)
    else:
        db_index = None

    # Calinski-Harabasz index
    if nb_clusters > 1:
        ch_index = calinski_harabasz_score(dataset, labels)
    else:
        ch_index = None

    print(f"silhouette: {silhouette:g}\ndb_index: {db_index:g}\nch_index: {ch_index:g}\n")

    return matricules, scores_SASI, nb_transactions, clusters, nb_clusters, cluster_sizes, silhouette, db_index, ch_index
	
# =================================================
# 2. Static PDF visualization of the dataset in R^2
# =================================================
def visualize_customers_pdf(nb_clusters, scores_SASI, clusters, cluster_sizes, date_start_str, date_end_str, silhouette, db_index, ch_index, file_pdf):

    # Extracting coordinates X and Y
    dataset = np.array(scores_SASI, dtype=float)
    X = dataset[:, 0]
    Y = dataset[:, 1]

    # Exact limits
    Xmin, Xmax = np.min(X) - 0.05, np.max(X) + 0.05
    Ymin, Ymax = np.min(Y) - 0.05, np.max(Y) + 0.05

    # name of each cluster
    cluster_name = [f"c{k+1}" for k in range(nb_clusters)]

    # Creation of the figure
    fig, ax = plt.subplots(figsize=(8,8))
	
    # Different markers per cluster
    markers = ["o","s","p","d","*","+","<",">","v"]

    # Displaying the scatter plot
    for k in np.unique(clusters):
        mask = clusters == k
        legend = f"{cluster_name[k]} -> {cluster_sizes[k]}"
		
        ax.scatter(
            X[mask],
            Y[mask],
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
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)

    # Legend
    ax.set_xlabel("Sequential Amounts (SA)")
    ax.set_ylabel("Sequential Intensity (SI)")
	
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,f"Analysis from {date_start_str} to {date_end_str}", ha="center", fontsize=10)
    fig.text(0.5, 0.02,f"silhouette: {silhouette:g},  dbi: {db_index:g},  chi: {ch_index:g}", ha="center", fontsize=10)
    
    ax.grid(True)
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
# 21. Dynamic HTML visualization of the dataset in R^2
# ====================================================
def visualize_customers_html(nb_clusters, scores_SASI, nb_transactions, clusters, cluster_sizes, matricules,
                             date_start_str, date_end_str, silhouette, db_index, ch_index, file_html):

    # Extracting coordinates X and Y
    dataset = np.array(scores_SASI, dtype=float)
    X = dataset[:, 0].tolist()
    Y = dataset[:, 1].tolist()

    # Cluster names
    cluster_name = [f"c{k+1}" for k in range(nb_clusters)]

    matricules = list(map(str, matricules))
    clusters = list(map(int, clusters))
    nb_transactions = list(map(int, nb_transactions))

    # Preparing data for JavaScript
    data_points = []
    for i in range(len(matricules)):
        data_points.append({
            "matricule": matricules[i],
            "X": X[i],
            "Y": Y[i],
            "cluster": clusters[i],
            "cluster_name": cluster_name[clusters[i]],
            "transactions": nb_transactions[i]
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
    gap: 65px;
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

    <!-- Search bar -->
    <div id="search-bar">
        <div>
            <input type="text" id="searchBox" placeholder="Matricule" style="padding:5px;">
            <button onclick="searchClient()">🔍</button>
        </div>
        <div id="result"></div>
    </div>

    <!-- Plot -->
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
            "<br>SA: " + p.X.toFixed(4) +
            "<br>SI: " + p.Y.toFixed(4)
        ),
        hoverinfo: "text"
    }});

    traceIndex++;
}});

// =======================
// Layout
// =======================
let layout = {{
    xaxis: {{title: "Sequential Amounts (SA)"}},
    yaxis: {{title: "Sequential Intensity (SI)"}},
    legend: {{
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: 1.1
    }},
    plot_bgcolor: "#eff6ff",
    hovermode: "closest",
    margin: {{t: 100, b: 120}},
    annotations: [
        {{
            text: "Analysis from {date_start_str} to {date_end_str}",
            x: 0.5,
            y: -0.12,
            xref: "paper",
            yref: "paper",
            showarrow: false
        }},
        {{
            text: "silhouette: {silhouette:g}, dbi: {db_index:g}, chi: {ch_index:g}",
            x: 0.5,
            y: -0.16,
            xref: "paper",
            yref: "paper",
            showarrow: false
        }}
    ]
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

        let A = found.X;
        let B = found.Y;

        res.innerHTML = found.cluster_name + " (" + A.toFixed(4) + " , " + B.toFixed(4) + ")";

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

    # Reading the parameters to be passed to the command line
    parser = argparse.ArgumentParser(description="2D visualization of customer segmentation")

    parser.add_argument("--csv_clustering", type=str, default="./clustering.csv", help="CSV file containing the results of the customer segmentation")
    parser.add_argument("--pdf_clustering", type=str, default="./clustering.pdf", help="Static PDF file containing the results of the customer segmentation")
    parser.add_argument("--html_clustering", type=str, default="./clustering.html", help="Dynamic HTML file containing the results of the customer segmentation")

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
    matricules, scores_SASI, nb_transactions, clusters, nb_clusters, cluster_sizes, silhouette, db_index, ch_index = read_clustering_results(csv_clustering)

    # Static PDF visualization of the dataset in R^2
    visualize_customers_pdf(nb_clusters, scores_SASI, clusters, cluster_sizes, date_start_str, date_end_str, silhouette, db_index, ch_index, pdf_clustering)

    # Dynamic HTML visualization of the dataset in R^2
    visualize_customers_html(nb_clusters, scores_SASI, nb_transactions, clusters, cluster_sizes, matricules, date_start_str, date_end_str, silhouette, db_index, ch_index, html_clustering)

    # Saving the metrics in a file
    with open("./metrics_customer_segmentation.txt", "w") as f:
        f.write(f"silhouette: {silhouette:g}\n")
        f.write(f"db_index: {db_index:g}\n")
        f.write(f"ch_index: {ch_index:g}\n")

    # End of the timing of the time cost
    end = time.perf_counter()
    print(f"\nTime complexity: {end - start:g} s")

if __name__ == "__main__":
    main()
	