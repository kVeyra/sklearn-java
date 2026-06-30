package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.math.RandomGenerator;
import org.sklearn.tree.DecisionTreeClassifier;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Random Forest classifier.
 *
 * <p>Fits an ensemble of decision tree classifiers on bootstrap samples
 * of the training data, using random feature selection at each split.
 * Predictions are averaged across trees (soft voting).
 *
 * <p>Mirrors {@code sklearn.ensemble.RandomForestClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * RandomForestClassifier rf = new RandomForestClassifier(100, 5, 2, 1);
 * rf.fit(X, y);
 * Vector preds = rf.predict(X_test);
 * }</pre>
 */
public class RandomForestClassifier implements Predictor<Matrix, Vector, Vector> {

    private List<DecisionTreeClassifier> trees;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private int maxFeatures;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;
    private long seed;

    /**
     * Create a random forest classifier.
     *
     * @param nEstimators      number of trees in the forest
     * @param maxDepth         maximum depth of each tree
     * @param minSamplesSplit  minimum samples to split
     * @param minSamplesLeaf   minimum samples at a leaf
     */
    public RandomForestClassifier(int nEstimators, int maxDepth,
                                  int minSamplesSplit, int minSamplesLeaf) {
        this(nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, 0, 42);
    }

    /**
     * Create a random forest classifier with full control.
     *
     * @param nEstimators      number of trees
     * @param maxDepth         max depth per tree
     * @param minSamplesSplit  min samples to split
     * @param minSamplesLeaf   min samples at leaf
     * @param maxFeatures      number of features to consider at each split
     *                         (0 = use sqrt(n_features))
     * @param seed             random seed for reproducibility
     */
    public RandomForestClassifier(int nEstimators, int maxDepth,
                                  int minSamplesSplit, int minSamplesLeaf,
                                  int maxFeatures, long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.seed = seed;
    }

    @Override
    public RandomForestClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) y.get(i));
        }
        this.classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);

        int effectiveMaxFeatures = maxFeatures > 0
            ? Math.min(maxFeatures, m)
            : Math.max(1, (int) Math.sqrt(m));

        RandomGenerator rng = new RandomGenerator(seed);
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
            // Bootstrap sample
            int[] bootstrapIdx = new int[n];
            for (int i = 0; i < n; i++) {
                bootstrapIdx[i] = rng.nextInt(n);
            }

            Matrix bootX = new Matrix(n, m);
            Vector bootY = new Vector(n);
            for (int i = 0; i < n; i++) {
                int idx = bootstrapIdx[i];
                for (int j = 0; j < m; j++) {
                    bootX.set(i, j, X.get(idx, j));
                }
                bootY.set(i, y.get(idx));
            }

            DecisionTreeClassifier tree = new DecisionTreeClassifier(
                maxDepth, minSamplesSplit, minSamplesLeaf, "gini");
            tree.fit(bootX, bootY);
            trees.add(tree);
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RandomForestClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        Vector probs = predictProba(X);
        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            preds[i] = probs.get(i) >= 0.5 ? classes[1] : classes[0];
        }
        return new Vector(preds);
    }

    /**
     * Predict class probabilities (positive class fraction across trees).
     */
    public Vector predictProba(Matrix X) {
        Validation.checkFitted(fitted, "RandomForestClassifier");
        Validation.checkMatrix(X, -1);

        int n = X.rows();
        double[] probs = new double[n];

        for (DecisionTreeClassifier tree : trees) {
            Vector treeProbs = tree.predictProba(X);
            for (int i = 0; i < n; i++) {
                probs[i] += treeProbs.get(i);
            }
        }

        for (int i = 0; i < n; i++) {
            probs[i] /= trees.size();
        }

        return new Vector(probs);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "RandomForestClassifier");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
    }

    public int[] getClasses() {
        return classes;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_estimators", nEstimators);
        params.put("max_depth", maxDepth);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        return Collections.unmodifiableMap(params);
    }
}
