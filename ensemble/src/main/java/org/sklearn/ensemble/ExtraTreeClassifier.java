package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;

import java.util.*;

/**
 * Extra-Tree classifier (extremely randomized decision tree).
 *
 * <p>A decision tree with random split thresholds, grown on the full
 * training data (no bootstrap).
 *
 * <p>Mirrors {@code sklearn.ensemble.ExtraTreeClassifier}.
 */
public class ExtraTreeClassifier implements Predictor<Matrix, Vector, Vector> {

    private DecisionTreeClassifier tree;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private String criterion;
    private long seed;

    public ExtraTreeClassifier() {
        this(10, 2, 1, "gini", 42);
    }

    public ExtraTreeClassifier(int maxDepth, int minSamplesSplit,
                                int minSamplesLeaf, String criterion, long seed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = criterion;
        this.seed = seed;
    }

    @Override
    public ExtraTreeClassifier fit(Matrix X, Vector y) {
        tree = new DecisionTreeClassifier(maxDepth, minSamplesSplit, minSamplesLeaf,
            criterion, true, seed);
        tree.fit(X, y);
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        return tree.predict(X);
    }

    public Vector predictProba(Matrix X) {
        return tree.predictProba(X);
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
        p.put("criterion", criterion);
        return p;
    }
}
