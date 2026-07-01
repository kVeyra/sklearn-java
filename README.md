# sklearn-java

[![Build](https://github.com/kVeyra/sklearn-java/actions/workflows/ci.yml/badge.svg)](https://github.com/kVeyra/sklearn-java/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/kVeyra/sklearn-java/branch/develop/graph/badge.svg)](https://codecov.io/gh/kVeyra/sklearn-java)
[![Javadoc](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kVeyra.github.io/sklearn-java)
[![Java](https://img.shields.io/badge/Java-21-blue)](https://openjdk.org/projects/jdk/21/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![GitHub Discussions](https://img.shields.io/badge/Ask%20us-anything-blue.svg)](https://github.com/kVeyra/sklearn-java/discussions)
[![GitHub stars](https://img.shields.io/github/stars/kVeyra/sklearn-java?style=social)](https://github.com/kVeyra/sklearn-java)

**A production-quality Java reimplementation of Python's scikit-learn.** Covers the core ML API surface (in progress) with behavioral compatibility, numerical accuracy to 1e-8, and pure Java — no Python, JNI, or NumPy.

## Why sklearn-java?

- **Drop-in replacement mindset** — API mirrors sklearn exactly: `.fit()`, `.predict()`, `.transform()`, `.score()`
- **Numerical accuracy** — every algorithm validated against Python sklearn with 1e-8 tolerance
- **Pure Java** — zero Python dependencies, zero JNI, zero external native libs
- **Modern Java** — Java 21, sealed interfaces, records, pattern matching
- **100% coverage goal** — targeting 400+ public classes matching sklearn's API surface

## Modules

| Module | Status | Coverage |
|--------|--------|----------|
| `math` | ✅ Complete | Vectors, dense matrices, random generators |
| `core` | ✅ Complete | Estimator/Predictor/Transformer interfaces, Pipeline |
| `preprocessing` | ✅ Complete | StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, PolynomialFeatures, Binarizer, KBinsDiscretizer, FunctionTransformer |
| `linear_model` | ✅ Complete | LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression |
| `tree` | ✅ Complete | DecisionTreeClassifier/Regressor (Gini, Entropy, MSE), random splits |
| `ensemble` | ✅ Complete | RandomForest, AdaBoost, GradientBoosting, Bagging, Voting, Stacking, IsolationForest, ExtraTrees, ExtraTreesEmbedding |
| `svm` | ✅ Complete | SVC (one-vs-one), SVR (ε-insensitive), linear/poly/RBF/sigmoid kernels |
| `naive_bayes` | ✅ Complete | GaussianNB |
| `neighbors` | ✅ Complete | KNeighborsClassifier, KNeighborsRegressor |
| `cluster` | ✅ Complete | KMeans (Lloyd's + k-means++), DBSCAN |
| `feature_selection` | ✅ Complete | VarianceThreshold |
| `decomposition` | ✅ Complete | PCA |
| `metrics` | ✅ Complete | ClassificationMetrics, RegressionMetrics, RankingMetrics, PairwiseMetrics, ClusteringMetrics |
| `model_selection` | ✅ Complete | KFold, StratifiedKFold, CrossValidation, GridSearchCV, TrainTestSplit |
| `ensemble` (phase A2) | ✅ Complete | GradientBoosting, Bagging, Voting, Stacking, IsolationForest, ExtraTree, RandomTreesEmbedding |
| `utils` | ✅ Complete | Validation, matrix/vector utilities |
| `datasets` | ❌ Not started | Toy datasets will be added in Phase C |

## Quick Start

```java
// Standardize features
StandardScaler scaler = new StandardScaler();
Matrix X_scaled = scaler.fitTransform(X_train);

// Train a classifier
RandomForestClassifier rf = new RandomForestClassifier(100, 5);
rf.fit(X_scaled, y_train);

// Predict & evaluate
Vector preds = rf.predict(scaler.transform(X_test));
double acc = ClassificationMetrics.accuracy(y_test, preds);
```

```java
// Grid search with cross-validation
GridSearchCV grid = new GridSearchCV(
    new SVC(),
    Map.of("C", new double[]{0.1, 1.0, 10.0}, "kernel", new String[]{"rbf", "linear"}),
    5
);
grid.fit(X_train, y_train);
System.out.println("Best params: " + grid.bestParams());
```

## Building

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
./gradlew build
```

## Testing

```bash
./gradlew test
./gradlew jacocoTestReport  # coverage report at build/reports/
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Branch strategy (feature branches → develop → main)
- Coding standards (Java 21, Google Java Format)
- Validation requirements (1e-8 tolerance against sklearn)
- PR checklist

**Looking for good first issues?** Check the [issue tracker](https://github.com/kVeyra/sklearn-java/issues) and the [COVERAGE_PLAN.md](COVERAGE_PLAN.md) for remaining algorithms.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| Phase A1 | Metrics + Model Selection | ✅ Complete |
| Phase A2 | Ensemble Methods | ✅ Complete |
| Phase A3 | Linear Models (SGD, RidgeCV, Bayesian, Huber) | 🔜 Next |
| Phase A4 | Naive Bayes + Neighbors (full) | 🔜 Planned |
| Phase A5 | Neural Network + Feature Selection | 🔜 Planned |
| Phase B | Manifold, Impute, Pipeline (full) | 📋 Planned |
| Phase C | Decomposition, Covariance, Cross-decomposition | 📋 Planned |
| Phase D | Semi-supervised, Multi-output, Niche estimators | 📋 Planned |

See [COVERAGE_PLAN.md](COVERAGE_PLAN.md) for full 400+ item breakdown.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Stats

![Alt](https://repobeats.axiom.co/api/placeholder.svg "Repo stats")
