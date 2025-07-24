# GNN-based-Link-Prediction-for-Personalized-Video-Recommendations-in-Bipartite-User-Item-Networks
This project explores the use of Graph Neural Networks (GNNs) for building a personalized video recommendation system, modeled as a link prediction problem on a bipartite graph. The system aims to predict which unseen movies a user is most likely to interact with, based on historical interaction data and the structure of the user–item graph.
We use the **MovieLens 100K** dataset, which consists of user ratings for movies, and construct a **user–movie bipartite graph** where:

- Each **node** is either a **user** or a **movie**.
- An **edge** between a user and a movie represents a **rating or interaction (positive signal)**.

---

## 🔧 Methodology

### 📊 Data Preprocessing:
- Ratings **≥ 4** are considered **positive edges (links)**.
- User and movie IDs are **label-encoded** to form a contiguous graph index.

### 🧱 Graph Construction:
- A **bipartite graph** is built using **PyTorch Geometric**.
- Edges are split into **train**, **validation**, and **test** sets for **link prediction**.

### 🧠 Model Architecture:
- A **two-layer GraphSAGE** model is used to generate low-dimensional **node embeddings**.
- A **dot-product decoder** computes the likelihood of user–movie links.
- The model is trained to distinguish between **real and fake links** using **binary cross-entropy loss**.

### 📈 Training & Evaluation:
- The model is trained for **200 epochs**.
- Evaluated using **Area Under the ROC Curve (AUC)** on the test set.
- Achieved **Test AUC ~0.88**, indicating strong link prediction capability.

### 🎯 Recommendation:
- For a given user, the trained model generates **top-N movie recommendations**.
- Movie IDs are **decoded back into human-readable titles** using MovieLens metadata.

### 🔍 Visualization:
- Sample **user–movie subgraphs** are visualized using **NetworkX** and **matplotlib** to demonstrate graph structure and interactions.
