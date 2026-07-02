package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Extra-Trees regressor (Extremely Randomized Trees).
 *
 * <p>Fits an ensemble of decision tree regressors using the full training set
 * (no bootstrap) and random split thresholds at each node.
 *
 * <p>Mirrors {@code sklearn.ensemble.ExtraTreesRegressor}.
 *
 * <p>Usage:
 * <pre>{@code
 * ExtraTreesRegressor et = new ExtraTreesRegressor(100, 5, 2, 1, 42);
 * et.fit(X, y);
 * Vector preds = et.predict(X_test);
 * }</pre>
 */
public class ExtraTreesRegressor implements Predictor<Matrix, Vector, Vector> {

    private List<DecisionTreeRegressor> trees;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean fitted;
    private int nFeatures;
    private long seed;

    /**
     * Create an Extra-Trees regressor.
     *
     * @param nEstimators      number of trees
     * @param maxDepth         maximum depth of each tree
     * @param minSamplesSplit  min samples to split
     * @param minSamplesLeaf   min samples at a leaf
     * @param seed             random seed
     */
    public ExtraTreesRegressor(int nEstimators, int maxDepth,
                                int minSamplesSplit, int minSamplesLeaf,
                                long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public ExtraTreesRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int m = X.cols();
        this.nFeatures = m;

        trees = new ArrayList<>();
        for (int t = 0; t < nEstimators; t++) {
            DecisionTreeRegressor tree = new DecisionTreeRegressor(
                maxDepth, minSamplesSplit, minSamplesLeaf, true, seed + t * 1000L);
            tree.fit(X, y);
            trees.add(tree);
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ExtraTreesRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];

        for (DecisionTreeRegressor tree : trees) {
            Vector treePred = tree.predict(X);
            for (int i = 0; i < n; i++) {
                preds[i] += treePred.get(i);
            }
        }
        for (int i = 0; i < n; i++) {
            preds[i] /= trees.size();
        }

        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "ExtraTreesRegressor");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        double ssRes = 0, ssTot = 0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0) {
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
