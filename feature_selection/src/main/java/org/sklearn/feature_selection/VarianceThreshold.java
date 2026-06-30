package org.sklearn.feature_selection;

import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Feature selector that removes features with variance below a threshold.
 *
 * <p>Mirrors {@code sklearn.feature_selection.VarianceThreshold}.
 *
 * <p>Usage:
 * <pre>{@code
 * VarianceThreshold selector = new VarianceThreshold(0.1);
 * selector.fit(X);
 * Matrix Xreduced = selector.transform(X);
 * }</pre>
 */
public class VarianceThreshold {

    private double threshold;
    private boolean fitted;
    private boolean[] selectedMask;
    private int[] selectedIndices;

    public VarianceThreshold() {
        this(0.0);
    }

    public VarianceThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * Learn which features to keep.
     */
    public VarianceThreshold fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();

        selectedMask = new boolean[m];
        List<Integer> keep = new ArrayList<>();

        for (int j = 0; j < m; j++) {
            double mean = 0.0;
            for (int i = 0; i < n; i++) {
                mean += X.get(i, j);
            }
            mean /= n;

            double var = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = X.get(i, j) - mean;
                var += diff * diff;
            }
            var /= n;

            if (threshold == 0.0) {
                selectedMask[j] = var > 0.0;
            } else {
                selectedMask[j] = var >= threshold;
            }
            if (selectedMask[j]) {
                keep.add(j);
            }
        }

        selectedIndices = keep.stream().mapToInt(Integer::intValue).toArray();
        fitted = true;
        return this;
    }

    /**
     * Apply dimensionality reduction to X.
     */
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "VarianceThreshold");
        Validation.checkMatrix(X, -1);
        if (X.cols() != selectedMask.length) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + selectedMask.length);
        }

        int n = X.rows();
        int m = selectedIndices.length;
        Matrix result = new Matrix(n, m);
        for (int j = 0; j < m; j++) {
            int srcCol = selectedIndices[j];
            for (int i = 0; i < n; i++) {
                result.set(i, j, X.get(i, srcCol));
            }
        }
        return result;
    }

    /**
     * Fit and transform in one call.
     */
    public Matrix fitTransform(Matrix X) {
        fit(X);
        return transform(X);
    }

    public double[] getVariances() {
        return null; // not stored
    }

    public boolean[] getSupport() {
        return selectedMask;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("threshold", threshold);
        return Collections.unmodifiableMap(params);
    }
}
