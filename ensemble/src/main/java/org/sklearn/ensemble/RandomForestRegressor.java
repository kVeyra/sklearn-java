package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.math.RandomGenerator;
import org.sklearn.tree.DecisionTreeRegressor;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Random Forest regressor.
 *
 * <p>Fits an ensemble of decision tree regressors on bootstrap samples
 * of the training data, using random feature selection at each split.
 * Predictions are averaged across trees.
 *
 * <p>Mirrors {@code sklearn.ensemble.RandomForestRegressor}.
 *
 * <p>Usage:
 * <pre>{@code
 * RandomForestRegressor rf = new RandomForestRegressor(100, 5, 2, 1);
 * rf.fit(X, y);
 * Vector preds = rf.predict(X_test);
 * }</pre>
 */
public class RandomForestRegressor implements Predictor<Matrix, Vector, Vector> {

    private List<DecisionTreeRegressor> trees;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean fitted;
    private int nFeatures;
    private long seed;

    public RandomForestRegressor(int nEstimators, int maxDepth,
                                 int minSamplesSplit, int minSamplesLeaf) {
        this(nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, 42);
    }

    public RandomForestRegressor(int nEstimators, int maxDepth,
                                 int minSamplesSplit, int minSamplesLeaf,
                                 long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public RandomForestRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        RandomGenerator rng = new RandomGenerator(seed);
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
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

            DecisionTreeRegressor tree = new DecisionTreeRegressor(
                maxDepth, minSamplesSplit, minSamplesLeaf);
            tree.fit(bootX, bootY);
            trees.add(tree);
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RandomForestRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];

        for (DecisionTreeRegressor tree : trees) {
            Vector treePreds = tree.predict(X);
            for (int i = 0; i < n; i++) {
                preds[i] += treePreds.get(i);
            }
        }

        for (int i = 0; i < n; i++) {
            preds[i] /= trees.size();
        }

        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "RandomForestRegressor");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        double ssRes = 0.0;
        double ssTot = 0.0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
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
