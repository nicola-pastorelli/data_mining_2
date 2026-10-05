# Exploration of Spotify tracks dataset - DM2

The project analyses the Spotify tracks dataset, covering data preparation, time series analysis, outlier detection, imbalanced learning, advanced classification and regression, and explainable AI.

---

## Data

| Dataset       | Content                                              | Size                               |
|---------------|------------------------------------------------------|------------------------------------|
| `tracks.csv`  | Audio features and metadata of tracks (114 genres)   | 109,547 rows × 34 columns          |
| `artists.csv` | Artist popularity, followers, genres                 | 30,141 rows × 5 columns            |
| Time series   | Spectral centroid of each song over time (20 genres) | 10,000 series × 1,280 observations |

---

## Pipeline

### 1. Data understanding & preparation
- Removed irrelevant and highly correlated features (13 kept).
- Created five new features: `season`, `decade`, `months from publication`, `century`, `single artist`.
- Merged tracks and artists, but dropped the merged dataset because it skewed the genre distribution.
- Prepared the time series with Min-Max scaling, moving average and approximations (DFT, PAA, SAX).

### 2. Time series analysis
- **Clustering:** k-means (DTW) and hierarchical (Euclidean), both with k = 2, visualized with PCA and SVD.
- **Classification (20 genres):** KNN, MiniRocket and shapelets.
- **Motifs & discords:** compared with the most representative shapelet of each genre.

### 3. Advanced data preprocessing
- **Outliers:** KNN, LOF and Isolation Forest. The 1,421 outliers found by all three (≈ 1%) were kept in the data.
- **Imbalanced learning:** century classification (91% / 9%) with a Decision Tree, using SMOTE, ADASYN and Random Undersampling.

### 4. Advanced ML & XAI
- **Classification (20 genres):** Logistic Regression, SVC, Random Forest, Bagging (KNN) and XGBoost, tuned with grid search and 5-fold CV, plus a manually tuned PyTorch Neural Network.
- **Regression (track popularity):** Support Vector Regressor and Gradient Boosting Regressor.
- **Explainability:** LIME applied to the SVC to understand why the `folk` genre is hard to classify.

---

## Key results

| Task                              | Best model                      | Result                              |
|-----------------------------------|---------------------------------|-------------------------------------|
| Time series genre classification  | KNN (Euclidean, DFT)            | Accuracy 0.23                       |
| Tabular genre classification      | Neural Network + extra features | Accuracy 0.63                       |
| Imbalanced century classification | Decision Tree + SMOTE           | Minority recall 0.62 (up from 0.44) |
| Popularity regression             | Gradient Boosting Regressor     | R² 0.482                            |

---

## Conclusions

- Time series genre classification is hard (accuracy ≤ 0.23, random baseline 0.05). Euclidean distance slightly outperformed DTW, probably because the series are already well aligned.
- Shapelets appear closer to motifs than to discords, suggesting that genres are better described by recurring patterns than by anomalies.
- Outliers are consistent across methods from different families. Among the outliers shared by all three methods, `sleep` is the most frequent genre.
- Oversampling improves minority-class recall at the cost of overall accuracy.
- Advanced models beat simple ones, and the engineered features improved the neural network.
- `folk` and `mpb` are the hardest genres to classify.

---

## Authors

Nicola Pastorelli, Salvatore Ergoli, Marco Sanna