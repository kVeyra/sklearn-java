package org.sklearn.tree;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class ExtraTreeRegressor implements Predictor<Matrix, Vector, Vector> {

    private DecisionTreeRegressor tree;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private long seed;
    private boolean fitted;

    public ExtraTreeRegressor() {
        this(Integer.MAX_VALUE, 2, 1, 42);
    }

    public ExtraTreeRegressor(int maxDepth, int minSamplesSplit, int minSamplesLeaf, long seed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public ExtraTreeRegressor fit(Matrix X, Vector y) {
        tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit, minSamplesLeaf, true, seed);
        tree.fit(X, y);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ExtraTreeRegressor");
        return tree.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "ExtraTreeRegressor");
        return tree.score(X, y);
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", maxDepth);
        p.put("min_samples_split", minSamplesSplit);
        p.put("min_samples_leaf", minSamplesLeaf);
        p.put("seed", seed);
        return Collections.unmodifiableMap(p);
    }
}
