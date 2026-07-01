package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;

import java.util.*;

/**
 * Extra-Tree regressor (extremely randomized decision tree).
 *
 * <p>A decision tree with random split thresholds, grown on the full
 * training data (no bootstrap).
 *
 * <p>Mirrors {@code sklearn.ensemble.ExtraTreeRegressor}.
 */
public class ExtraTreeRegressor implements Predictor<Matrix, Vector, Vector> {

    private DecisionTreeRegressor tree;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private long seed;

    public ExtraTreeRegressor() {
        this(10, 2, 1, 42);
    }

    public ExtraTreeRegressor(int maxDepth, int minSamplesSplit,
                               int minSamplesLeaf, long seed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public ExtraTreeRegressor fit(Matrix X, Vector y) {
        tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit, minSamplesLeaf, true, seed);
        tree.fit(X, y);
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        return tree.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return tree.score(X, y);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", maxDepth);
        p.put("min_samples_split", minSamplesSplit);
        p.put("min_samples_leaf", minSamplesLeaf);
        return p;
    }
}
