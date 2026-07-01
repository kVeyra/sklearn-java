# 100% Coverage Plan: sklearn-java

## Current Status: ~40/400+ (~10%) — 19 submodules, 95 tests

## Phase A — Core ML (High Impact, Most-Used APIs)

### A1. Metrics + Model Selection
- **Metrics**: `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `r2_score`, `mean_absolute_percentage_error`, `median_absolute_error`, `max_error`, `explained_variance_score` — all regression metrics
- `roc_curve`, `roc_auc_score`, `auc`, `average_precision_score`, `precision_recall_curve`, `det_curve` — ranking/classification metrics  
- `pairwise_distances`, `euclidean_distances`, `pairwise_kernels`, `nan_euclidean_distances` — pairwise metrics
- `silhouette_score`, `silhouette_samples`, `calinski_harabasz_score`, `davies_bouldin_score`, `adjusted_rand_score`, `mutual_info_score`, `adjusted_mutual_info_score`, `normalized_mutual_info_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`, `fowlkes_mallows_score`, `rand_score` — clustering metrics
- `log_loss`, `hinge_loss`, `brier_score_loss`, `cohen_kappa_score`, `matthews_corrcoef`, `balanced_accuracy_score`, `hamming_loss`, `zero_one_loss`, `top_k_accuracy_score`, `jaccard_score`, `fbeta_score`, `classification_report` — classification metrics
- **Model Selection**: `KFold`, `StratifiedKFold`, `GroupKFold`, `LeaveOneOut`, `ShuffleSplit`, `StratifiedShuffleSplit`, `TimeSeriesSplit`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `GroupShuffleSplit`, `PredefinedSplit`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `LeavePOut`
- `cross_val_score`, `cross_validate`, `cross_val_predict`
- `GridSearchCV`, `RandomizedSearchCV`, `ParameterGrid`, `ParameterSampler`
- `learning_curve`, `validation_curve`, `permutation_test_score`
- `check_cv`, `make_scorer`, `get_scorer`

### A2. Ensemble (Remaining) + Tree Extras
- `GradientBoostingClassifier`, `GradientBoostingRegressor`
- `BaggingClassifier`, `BaggingRegressor`
- `VotingClassifier`, `VotingRegressor`
- `StackingClassifier`, `StackingRegressor`
- `IsolationForest`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `RandomTreesEmbedding`
- `ExtraTreeClassifier`, `ExtraTreeRegressor` (standalone)

### A3. Linear Model (Remaining)
- `SGDClassifier`, `SGDRegressor`, `SGDOneClassSVM`
- `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`
- `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC`
- `ElasticNetCV`, `MultiTaskElasticNet`, `MultiTaskElasticNetCV`, `MultiTaskLasso`, `MultiTaskLassoCV`
- `BayesianRidge`, `ARDRegression`
- `HuberRegressor`, `RANSACRegressor`, `TheilSenRegressor`, `QuantileRegressor`
- `Perceptron`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`
- `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`
- `OrthogonalMatchingPursuit`, `OrthogonalMatchingPursuitCV`
- `LogisticRegressionCV`, `Lars`, `LarsCV`
- Functions: `enet_path`, `lars_path`, `lasso_path`, `ridge_regression`

### A4. Naive Bayes + Neighbors (Remaining)
- `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `CategoricalNB`
- `NearestNeighbors`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`
- `LocalOutlierFactor`, `NearestCentroid`, `KernelDensity`
- `NeighborhoodComponentsAnalysis`, `KNeighborsTransformer`, `RadiusNeighborsTransformer`
- Functions: `kneighbors_graph`, `radius_neighbors_graph`

### A5. Neural Network + Feature Selection
- `MLPClassifier`, `MLPRegressor`, `BernoulliRBM`
- `SelectKBest`, `SelectPercentile`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `GenericUnivariateSelect`
- `RFE`, `RFECV`, `SelectFromModel`, `SequentialFeatureSelector`
- Functions: `chi2`, `f_classif`, `f_regression`, `mutual_info_classif`, `mutual_info_regression`

## Phase B — ML Workbench (Medium Impact)

### B1. Decomposition (Remaining)
- `NMF`, `MiniBatchNMF`, `FastICA`, `TruncatedSVD`, `KernelPCA`, `IncrementalPCA`, `FactorAnalysis`
- `DictionaryLearning`, `MiniBatchDictionaryLearning`, `SparsePCA`, `MiniBatchSparsePCA`, `SparseCoder`
- `LatentDirichletAllocation`
- Functions: `fastica`, `dict_learning`, `non_negative_factorization`, `randomized_svd`

### B2. Clustering (Remaining)
- `AgglomerativeClustering`, `FeatureAgglomeration`, `Birch`, `OPTICS`, `SpectralClustering`
- `MeanShift`, `AffinityPropagation`, `MiniBatchKMeans`, `BisectingKMeans`, `HDBSCAN`
- `SpectralBiclustering`, `SpectralCoclustering`
- Functions: `k_means`, `mean_shift`, `affinity_propagation`, `spectral_clustering`, `dbscan`, `estimate_bandwidth`, `ward_tree`, `linkage_tree`

### B3. Impute + Datasets
- `SimpleImputer`, `KNNImputer`, `IterativeImputer`
- `load_iris`, `load_breast_cancer`, `load_digits`, `load_diabetes`, `load_wine`, `load_linnerud`
- `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_friedman1`, `make_low_rank_matrix`, `make_spd_matrix`, `make_swiss_roll`, `make_s_curve`, `make_gaussian_quantiles`, `make_multilabel_classification`, `make_biclusters`, `make_checkerboard`, `make_sparse_spd_matrix`, `make_sparse_coded_signal`, `make_sparse_uncorrelated`, `make_hastie_10_2`

### B4. SVM Remaining + Tree Extras
- `NuSVC`, `NuSVR`, `LinearSVC`, `LinearSVR`, `OneClassSVM`
- `l1_min_c` function
- `ExtraTreeClassifier`, `ExtraTreeRegressor` (standalone from Ensemble)

## Phase C — Specialized Modules (Niche Use Cases)

### C1. Manifold Learning
- `TSNE`, `Isomap`, `MDS`, `ClassicalMDS`, `SpectralEmbedding`, `LocallyLinearEmbedding`
- Functions: `locally_linear_embedding`, `smacof`, `spectral_embedding`, `trustworthiness`

### C2. Gaussian Process, Mixture, Cross Decomposition, LDA/QDA
- `GaussianProcessRegressor`, `GaussianProcessClassifier`, kernels
- `GaussianMixture`, `BayesianGaussianMixture`
- `PLSRegression`, `PLSCanonical`, `CCA`, `PLSSVD`
- `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`

### C3. Multiclass, MultiOutput, Kernel Methods, Semi-Supervised
- `OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier`
- `MultiOutputRegressor`, `MultiOutputClassifier`, `ClassifierChain`, `RegressorChain`
- `KernelRidge`, `RBFSampler`, `SkewedChi2Sampler`, `AdditiveChi2Sampler`, `Nystroem`, `PolynomialCountSketch`
- `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`

### C4. Covariance, Feature Extraction, Isotonic, Pipeline Extras
- `EmpiricalCovariance`, `LedoitWolf`, `ShrunkCovariance`, `MinCovDet`, `GraphicalLasso`, `GraphicalLassoCV`, `OAS`, `EllipticEnvelope`
- `DictVectorizer`, `FeatureHasher`, `CountVectorizer`, `TfidfVectorizer`, `TfidfTransformer`
- `IsotonicRegression`
- `FeatureUnion`, `ColumnTransformer`

## Execution Strategy
- Each phase → feature branch → PR → squash-merge to develop
- Each algorithm = class + JUnit 5 tests + checkstyle clean
- Deterministic: same seed → same results
- Javadoc on every public class
- Pipeline compatibility from day one
