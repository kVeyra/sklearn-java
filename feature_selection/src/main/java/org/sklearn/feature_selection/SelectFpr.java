package org.sklearn.feature_selection;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Feature selector that selects features based on a false positive rate (FPR) test.
 *
 * <p>Selects features whose p-values are below a threshold alpha.
 *
 * <p>Mirrors {@code sklearn.feature_selection.SelectFpr}.
 */
public class SelectFpr implements Transformer<Matrix, Matrix> {

    private double alpha;
    private String scoreFunc;
    private boolean fitted;
    private int[] selectedIndices;
    private int nFeatures;

    public SelectFpr() {
        this(0.05, "f_classif");
    }

    public SelectFpr(double alpha, String scoreFunc) {
        this.alpha = alpha;
        this.scoreFunc = scoreFunc;
    }

    @Override
    public SelectFpr fit(Matrix X, Matrix y) {
        return fit(X, y.col(0));
    }

    public SelectFpr fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nFeatures = X.cols();

        FScoring.FTestResult result;
        if ("f_classif".equals(scoreFunc)) {
            result = FScoring.fClassif(X, y);
        } else if ("f_regression".equals(scoreFunc)) {
            result = FScoring.fRegression(X, y);
        } else {
            throw new IllegalArgumentException("Unsupported score function: " + scoreFunc);
        }

        List<Integer> selected = new ArrayList<>();
        for (int j = 0; j < nFeatures; j++) {
            if (result.pvalue[j] <= alpha) {
                selected.add(j);
            }
        }

        selectedIndices = selected.stream().mapToInt(Integer::intValue).toArray();
        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SelectFpr");
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
        throw new UnsupportedOperationException("SelectFpr does not support inverseTransform");
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
        p.put("alpha", alpha);
        p.put("score_func", scoreFunc);
        return Collections.unmodifiableMap(p);
    }
}
