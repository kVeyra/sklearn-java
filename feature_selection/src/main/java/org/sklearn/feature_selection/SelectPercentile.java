package org.sklearn.feature_selection;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class SelectPercentile implements Transformer<Matrix, Matrix> {

    private double percentile;
    private String scoreFunc;
    private boolean fitted;
    private int[] selectedIndices;
    private int nFeatures;

    public SelectPercentile() {
        this(10, "f_classif");
    }

    public SelectPercentile(double percentile, String scoreFunc) {
        this.percentile = percentile;
        this.scoreFunc = scoreFunc;
    }

    @Override
    public SelectPercentile fit(Matrix X, Matrix y) {
        return fit(X, y.col(0));
    }

    public SelectPercentile fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        double[] scores;
        if ("f_classif".equals(scoreFunc)) {
            scores = FScoring.fClassif(X, y).statistic;
        } else if ("f_regression".equals(scoreFunc)) {
            scores = FScoring.fRegression(X, y).statistic;
        } else {
            throw new IllegalArgumentException("Unsupported score function: " + scoreFunc);
        }

        int nSelect = Math.max(1, (int) Math.ceil(nFeatures * percentile / 100.0));
        Integer[] idx = new Integer[nFeatures];
        for (int i = 0; i < nFeatures; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(scores[b], scores[a]));

        selectedIndices = new int[Math.min(nSelect, nFeatures)];
        for (int i = 0; i < selectedIndices.length; i++) selectedIndices[i] = idx[i];
        Arrays.sort(selectedIndices);

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SelectPercentile");
        Matrix result = new Matrix(X.rows(), selectedIndices.length);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                result.set(i, j, X.get(i, selectedIndices[j]));
            }
        }
        return result;
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("SelectPercentile does not support inverseTransform");
    }

    @Override
    public Matrix fitTransform(Matrix X, Matrix y) {
        return fit(X, y).transform(X);
    }

    public Matrix fitTransform(Matrix X, Vector y) {
        return fit(X, y).transform(X);
    }

    public int[] getSupport() { return selectedIndices; }
    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("percentile", percentile);
        p.put("score_func", scoreFunc);
        return Collections.unmodifiableMap(p);
    }
}
