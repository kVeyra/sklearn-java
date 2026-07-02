package org.sklearn.tree;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class ExtraTreeClassifier implements Predictor<Matrix, Vector, Vector> {

    private DecisionTreeClassifier tree;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private String criterion;
    private long seed;
    private boolean fitted;

    public ExtraTreeClassifier() {
        this(Integer.MAX_VALUE, 2, 1, "gini", 42);
    }

    public ExtraTreeClassifier(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                                String criterion, long seed) {
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
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ExtraTreeClassifier");
        return tree.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "ExtraTreeClassifier");
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
        p.put("criterion", criterion);
        p.put("seed", seed);
        return Collections.unmodifiableMap(p);
    }
}
