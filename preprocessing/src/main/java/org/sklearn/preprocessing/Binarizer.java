package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Binarize data (set feature values to 0 or 1) according to a threshold.
 *
 * <p>Values greater than the threshold map to 1, values less than or equal
 * to the threshold map to 0.
 *
 * <p>Mirrors {@code sklearn.preprocessing.Binarizer}.
 */
public class Binarizer implements Transformer<Matrix, Void> {

    private double threshold;
    private boolean fitted;

    /**
     * Create a Binarizer with the default threshold of 0.0.
     */
    public Binarizer() {
        this(0.0);
    }

    /**
     * Create a Binarizer with the specified threshold.
     *
     * @param threshold feature values below or equal to this become 0,
     *                  above become 1
     */
    public Binarizer(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public Binarizer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "Binarizer");
        Validation.checkMatrix(X, -1);

        int n = X.rows(), m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) > threshold ? 1.0 : 0.0;
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException(
            "Binarizer does not support inverse_transform");
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("threshold", threshold);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }
}
