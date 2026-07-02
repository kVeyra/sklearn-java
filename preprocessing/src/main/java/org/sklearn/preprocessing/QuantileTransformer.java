package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * QuantileTransformer transforms features to follow a uniform or normal distribution.
 *
 * <p>Mirrors {@code sklearn.preprocessing.QuantileTransformer}.
 */
public class QuantileTransformer implements Transformer<Matrix, Void> {

    private int nQuantiles;
    private String outputDistribution;
    private boolean fitted;
    private int nFeatures;
    private List<double[]> quantiles;

    /**
     * Create QuantileTransformer.
     *
     * @param nQuantiles          number of quantiles to compute
     * @param outputDistribution  "uniform" or "normal"
     */
    public QuantileTransformer(int nQuantiles, String outputDistribution) {
        this.nQuantiles = nQuantiles;
        this.outputDistribution = outputDistribution;
    }

    @Override
    public QuantileTransformer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        quantiles = new ArrayList<>();
        for (int j = 0; j < m; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X.get(i, j);
            Arrays.sort(col);

            int nq = Math.min(nQuantiles, n);
            double[] q = new double[nq];
            for (int k = 0; k < nq; k++) {
                int idx = (int) Math.round((double) k * (n - 1) / (nq - 1));
                q[k] = col[idx];
            }
            quantiles.add(q);
        }

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "QuantileTransformer");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];

        for (int j = 0; j < m; j++) {
            double[] q = quantiles.get(j);
            int nq = q.length;
            for (int i = 0; i < n; i++) {
                double val = X.get(i, j);
                int idx = Arrays.binarySearch(q, val);
                if (idx < 0) idx = -idx - 1;
                idx = Math.max(0, Math.min(idx, nq - 1));
                double uniform = (double) idx / (nq - 1);

                if (outputDistribution.equals("normal")) {
                    result[i][j] = inverseNormalCdf(uniform);
                } else {
                    result[i][j] = uniform;
                }
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("inverseTransform not supported for QuantileTransformer");
    }

    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_quantiles", nQuantiles);
        params.put("output_distribution", outputDistribution);
        return params;
    }

    private static double inverseNormalCdf(double p) {
        if (p <= 0) return -8;
        if (p >= 1) return 8;
        return 1.4142135623730951 * erfInv(2 * p - 1);
    }

    private static double erfInv(double x) {
        double w = -Math.log((1 - x) * (1 + x));
        double p;
        if (w < 5) {
            w = w - 2.5;
            p = 2.81022636e-08;
            p = 3.43273939e-07 + p * w;
            p = -3.5233877e-06 + p * w;
            p = -4.39150654e-06 + p * w;
            p = 0.00021858087 + p * w;
            p = -0.00125372503 + p * w;
            p = -0.00417768164 + p * w;
            p = 0.246640727 + p * w;
            p = 1.50140941 + p * w;
        } else {
            w = Math.sqrt(w) - 3;
            p = -0.000200214257;
            p = 0.000100950558 + p * w;
            p = 0.00134934322 + p * w;
            p = -0.00367342844 + p * w;
            p = 0.00573950773 + p * w;
            p = -0.0076224613 + p * w;
            p = 0.00943887047 + p * w;
            p = 1.00167406 + p * w;
            p = 2.83297682 + p * w;
        }
        return p * x;
    }
}
