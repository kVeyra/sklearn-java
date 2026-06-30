package org.sklearn.utils;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

/**
 * Input validation utilities mirroring sklearn's {@code check_array},
 * {@code check_X_y}, and related functions.
 */
public final class Validation {

    private Validation() {
    }

    /**
     * Validate that a matrix has the expected number of features.
     *
     * @param X        the data matrix
     * @param nFeatures expected number of features, or -1 to skip
     * @return the input matrix if valid
     */
    public static Matrix checkMatrix(Matrix X, int nFeatures) {
        if (X == null) {
            throw new IllegalArgumentException("Input data X must not be null");
        }
        if (X.rows() == 0) {
            throw new IllegalArgumentException("Input data X has zero rows");
        }
        if (nFeatures > 0 && X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Expected " + nFeatures + " features, got " + X.cols());
        }
        return X;
    }

    /**
     * Validate a target vector.
     *
     * @param y    the target vector
     * @param nSamples expected number of samples, or -1 to skip
     * @return the input vector if valid
     */
    public static Vector checkTarget(Vector y, int nSamples) {
        if (y == null) {
            throw new IllegalArgumentException("Target y must not be null");
        }
        if (y.size() == 0) {
            throw new IllegalArgumentException("Target y is empty");
        }
        if (nSamples > 0 && y.size() != nSamples) {
            throw new IllegalArgumentException(
                "Expected " + nSamples + " samples, got " + y.size());
        }
        return y;
    }

    /**
     * Validate an int array target.
     *
     * @param y    the target array
     * @param nSamples expected number of samples
     * @return the input array if valid
     */
    public static int[] checkLabels(int[] y, int nSamples) {
        if (y == null) {
            throw new IllegalArgumentException("Labels y must not be null");
        }
        if (nSamples > 0 && y.length != nSamples) {
            throw new IllegalArgumentException(
                "Expected " + nSamples + " samples, got " + y.length);
        }
        return y;
    }

    /**
     * Validate classification labels: must be 0..(nClasses-1).
     */
    public static void checkLabelsRange(int[] y, int nClasses) {
        for (int i = 0; i < y.length; i++) {
            if (y[i] < 0 || y[i] >= nClasses) {
                throw new IllegalArgumentException(
                    "Label at index " + i + " is " + y[i]
                        + ", expected 0.." + (nClasses - 1));
            }
        }
    }

    /**
     * Ensure the estimator is fitted before predict/transform.
     */
    public static void checkFitted(boolean isFitted, String name) {
        if (!isFitted) {
            throw new IllegalStateException(
                name + " is not fitted. Call fit() before using this estimator.");
        }
    }
}
