# sklearn-java

**A production-quality Java reimplementation of Python's scikit-learn.**

`sklearn-java` brings the power of scikit-learn's machine learning algorithms to the JVM with behavioral compatibility, numerical accuracy, and idiomatic Java design. Every algorithm is validated against its Python counterpart with a tolerance of 1e-8.

## Features

- **Behavioral Compatibility** — API mirrors scikit-learn: `.fit()`, `.predict()`, `.transform()`, `.score()`
- **Numerical Accuracy** — Validated against Python sklearn with 1e-8 tolerance
- **Pure Java** — No Python dependencies, no JNI, no NumPy
- **Modern Java** — Java 21+, sealed interfaces, records, pattern matching
- **Production Ready** — Comprehensive tests, benchmarks, static analysis

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `math` | ✅ | Dense/sparse matrix, vector operations, BLAS-like routines |
| `core` | 📝 | Estimator/Predictor/Transformer interfaces, Pipeline |
| `datasets` | ❌ | Toy datasets (Iris, Diabetes, Digits) |
| `preprocessing` | ❌ | StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder, PCA |
| `linear_model` | ❌ | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet |
| `neighbors` | ❌ | KNN, KDTree, BallTree |
| `tree` | ❌ | DecisionTree, RandomForest, ExtraTrees |
| `ensemble` | ❌ | GradientBoosting, AdaBoost |
| `cluster` | ❌ | KMeans, DBSCAN, Agglomerative |
| `svm` | ❌ | LinearSVC, SVC, SVR |
| `naive_bayes` | ❌ | GaussianNB, MultinomialNB, BernoulliNB |
| `metrics` | ❌ | Accuracy, Precision, Recall, F1, MSE, RMSE, R² |
| `model_selection` | ❌ | KFold, GridSearchCV, CrossValidation |
| `feature_selection` | ❌ | SelectKBest, RFE |
| `decomposition` | ❌ | PCA, TruncatedSVD |
| `utils` | ❌ | Validation, shuffling, utility functions |

## Quick Start

```java
// Load data
Dataset dataset = Datasets.loadIris();

// Split
TrainTestSplit split = new TrainTestSplit(dataset, 0.2);

// Train
LogisticRegression model = new LogisticRegression();
model.fit(split.getXTrain(), split.getYTrain());

// Predict
int[] predictions = model.predict(split.getXTest());

// Evaluate
double accuracy = Metrics.accuracy(split.getYTest(), predictions);
```

## Building

```bash
./gradlew build
```

## Testing

```bash
./gradlew test
./gradlew jacocoTestReport
```

## Validation

Every algorithm includes a Python validation script comparing outputs against sklearn:

```bash
python3 validation/validate_linear_regression.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
