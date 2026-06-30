package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Extra-Trees classifier (Extremely Randomized Trees).
 *
 * <p>Fits an ensemble of decision trees using the full training set
 * (no bootstrap) and random split thresholds at each node. The
 * randomization of both feature and threshold yields additional
 * diversity compared to RandomForest.
 *
 * <p>Mirrors {@code sklearn.ensemble.ExtraTreesClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * ExtraTreesClassifier et = new ExtraTreesClassifier(100, 5, 2, 1, 42);
 * et.fit(X, y);
 * Vector preds = et.predict(X_test);
 * }</pre>
 */
public class ExtraTreesClassifier implements Predictor<Matrix, Vector, Vector> {

    private List<DecisionTreeClassifier> trees;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;
    private long seed;

    /**
     * Create an Extra-Trees classifier.
     *
     * @param nEstimators      number of trees
     * @param maxDepth         maximum depth of each tree
     * @param minSamplesSplit  min samples to split
     * @param minSamplesLeaf   min samples at a leaf
     * @param seed             random seed
     */
    public ExtraTreesClassifier(int nEstimators, int maxDepth,
                                 int minSamplesSplit, int minSamplesLeaf,
                                 long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public ExtraTreesClassifier fit(Matrix X, Vector y) {
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

        RandomGenerator rng = new RandomGenerator(seed);
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
            DecisionTreeClassifier tree = new DecisionTreeClassifier(
                maxDepth, minSamplesSplit, minSamplesLeaf, "gini", true, seed + t * 1000L);
            tree.fit(X, y);
            trees.add(tree);
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "ExtraTreesClassifier");
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
        Validation.checkFitted(fitted, "ExtraTreesClassifier");
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
        Validation.checkFitted(fitted, "ExtraTreesClassifier");
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
