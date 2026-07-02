package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Center a kernel matrix (make it zero-mean in feature space).
 *
 * <p>Mirrors {@code sklearn.preprocessing.KernelCenterer}.
 */
public class KernelCenterer implements Transformer<Matrix, Void> {

    private boolean fitted;
    private double[] kMeanRows;
    private double kMeanAll;

    @Override
    public KernelCenterer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();

        kMeanRows = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < m; j++) sum += X.get(i, j);
            kMeanRows[i] = sum / m;
        }

        double sumAll = 0;
        for (int i = 0; i < n; i++) sumAll += kMeanRows[i];
        kMeanAll = sumAll / n;

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "KernelCenterer");
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            double rowMean = 0;
            for (int j = 0; j < m; j++) rowMean += X.get(i, j);
            rowMean /= m;
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) - rowMean - kMeanRows[j] + kMeanAll;
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("inverseTransform not supported for KernelCenterer");
    }

    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("K_fit_rows_", kMeanRows);
        params.put("K_fit_all_", kMeanAll);
        return params;
    }
}
