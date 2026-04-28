# Customer Segmentation

An unsupervised machine learning program that automatically groups mall customers into distinct segments based on their annual income and spending behaviour. Unlike supervised learning, there are no pre-defined labels, the model discovers the natural groupings hidden within the data on its own.

---

## What the Program Does

1. Loads a dataset of 200 mall customers with their demographic and spending details
2. Cleans and prepares the data (encodes gender, drops irrelevant columns, scales features)
3. Runs the Elbow Method to determine the optimal number of clusters (K)
4. Trains a K-Means Clustering model with K=5
5. Assigns every customer to one of the 5 discovered segments
6. Visualizes the clusters on scatter and pair plots
7. Summarizes each cluster with its average age, income, spending score, and a segment label
8. Saves all results to `results.txt` and all charts as PNG files

---

## Dataset

- **Source**: [Kaggle — Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **File**: `Mall_Customers.csv`
- **Records**: 200 customers
- **Columns**: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- **Features used for clustering**: Annual Income (k$), Spending Score (1-100)

---

## Concepts Used

### K-Means Clustering
K-Means is an unsupervised algorithm that groups data points into K clusters. It works by:

1. Placing K centroids (cluster centres) at random positions
2. Assigning each customer to the nearest centroid based on distance
3. Moving each centroid to the average position of all customers assigned to it
4. Repeating steps 2–3 until the assignments no longer change

The result is K groups where customers within the same group are as similar as possible, and customers in different groups are as distinct as possible. No labels are needed, the model discovers structure in the data on its own.

### Elbow Method
K-Means requires the number of clusters K to be chosen upfront. The Elbow Method helps find the best K by:

- Running K-Means for K = 1 through 10
- Recording the **inertia** for each K, the total sum of squared distances from each customer to their cluster centre
- Plotting inertia vs K and looking for the "elbow", the point where the curve bends and adding more clusters gives diminishing returns

For this dataset the elbow falls at **K = 5**, meaning 5 clusters capture the main patterns without over-segmenting.

### Feature Scaling
K-Means measures similarity using distance. If one feature has a much larger numerical range than another (e.g. income 15–137 vs spending score 1–99), it will dominate the distance calculations unfairly. `StandardScaler` transforms each feature so that it has:
- **Mean = 0**
- **Standard deviation = 1**

This puts both features on equal footing before clustering.

### Data Visualization
Charts make cluster patterns immediately understandable. Two types of charts are produced:

- **Scatter plot** — plots every customer as a dot (income on x-axis, spending score on y-axis), coloured by cluster, with centroids marked as X. Instantly shows the 5 distinct customer groups.
- **Pairplot** — shows scatter plots for every combination of features side by side. Useful for spotting which feature pairs separate clusters the most clearly.

---

## Clusters Discovered

| Cluster | Count | Avg Income | Avg Spend | Segment Label |
|---|---|---|---|---|
| 0 | 81 | $55k | 49.5 | Average / Mixed |
| 1 | 39 | $87k | 82.1 | High Income, High Spenders |
| 2 | 22 | $26k | 79.4 | Low Income, High Spenders |
| 3 | 35 | $88k | 17.1 | High Income, Low Spenders |
| 4 | 23 | $26k | 20.9 | Low Income, Low Spenders |

**Cluster 1** (High Income, High Spenders) is the most valuable customer segment -> high earners who spend freely.
**Cluster 3** (High Income, Low Spenders) represents cautious savers -> high earners who hold back on spending.
**Cluster 2** (Low Income, High Spenders) are likely younger impulse buyers spending beyond their means.

---

## Libraries and Their Purpose

| Library | Purpose |
|---|---|
| `pandas` | Load and manipulate the CSV dataset as a structured table (DataFrame). Used for reading data, encoding the Gender column, dropping unused columns, and computing per-cluster summaries with `groupby`. |
| `scikit-learn` (`sklearn`) | Provides `KMeans` for clustering, and `StandardScaler` for feature scaling so that both features contribute equally to distance calculations. |
| `matplotlib` | The foundational Python plotting library. Used to create and save all charts to PNG files. Set to `Agg` backend so charts are saved to files rather than opening in a window. |
| `seaborn` | Built on top of Matplotlib, provides higher-level, visually polished statistical charts with less code. Used for the `scatterplot` (clusters) and `pairplot` (all feature combinations). |
| `datetime` | Used to timestamp the `results.txt` output file with the date and time the program was run. |

---

## Output Files

| File | Description |
|---|---|
| `elbow_chart.png` | Inertia vs K plot — shows the elbow bend at K=5 |
| `clusters_income_vs_spending.png` | Scatter plot of all customers coloured by cluster, with centroids marked |
| `pairplot_clusters.png` | All feature combinations plotted against each other, coloured by cluster |
| `results.txt` | Full run report, dataset summary, elbow values, cluster summary table |

---

## How to Run

```bash
python segmentation.py
```

No user input required. The program runs fully automatically, prints the cluster summary to the console, and saves all output files.
