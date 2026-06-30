package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Bin continuous data into intervals.
 *
 * <p>Discretizes continuous features into k bins with either uniform
 * width or quantile-based bin edges.
 *
 * <p>Mirrors {@code sklearn.preprocessing.KBinsDiscretizer}.
 */
public class KBinsDiscretizer implements Transformer<Matrix, Void> {

    private int nBins;
    private String strategy;
    private boolean encodeOrdinal;
    private List<double[]> binEdges;
    private boolean fitted;

    /**
     * Create KBinsDiscretizer.
     *
     * @param nBins         number of bins per feature
     * @param strategy      "uniform" or "quantile"
     * @param encodeOrdinal if true, output ordinal integer codes;
     *                      if false, output one-hot encoded columns
     */
    public KBinsDiscretizer(int nBins, String strategy, boolean encodeOrdinal) {
        if (nBins < 2) {
            throw new IllegalArgumentException("nBins must be >= 2");
        }
        if (!strategy.equals("uniform") && !strategy.equals("quantile")) {
            throw new IllegalArgumentException(
                "strategy must be 'uniform' or 'quantile', got: " + strategy);
        }
        this.nBins = nBins;
        this.strategy = strategy;
        this.encodeOrdinal = encodeOrdinal;
    }

    @Override
    public KBinsDiscretizer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int m = X.cols();
        binEdges = new ArrayList<>();

        for (int j = 0; j < m; j++) {
            double[] col = new double[X.rows()];
            for (int i = 0; i < X.rows(); i++) {
                col[i] = X.get(i, j);
            }

            double[] edges;
            if (strategy.equals("uniform")) {
                double minVal = Double.POSITIVE_INFINITY;
                double maxVal = Double.NEGATIVE_INFINITY;
                for (double v : col) {
                    if (v < minVal) {
                        minVal = v;
                    }
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
                if (minVal == maxVal) {
                    edges = new double[]{minVal, minVal + 1.0};
                } else {
                    edges = new double[nBins + 1];
                    for (int k = 0; k <= nBins; k++) {
                        edges[k] = minVal + (maxVal - minVal) * k / nBins;
                    }
                }
            } else {
                Arrays.sort(col);
                edges = new double[nBins + 1];
                for (int k = 0; k <= nBins; k++) {
                    double p = (double) k / nBins;
                    double idx = p * (col.length - 1);
                    int lo = (int) Math.floor(idx);
                    int hi = (int) Math.ceil(idx);
                    if (lo == hi || lo >= col.length - 1) {
                        edges[k] = col[Math.min(lo, col.length - 1)];
                    } else {
                        edges[k] = col[lo] + (idx - lo) * (col[hi] - col[lo]);
                    }
                }
                edges[0] = col[0];
                edges[nBins] = col[col.length - 1];
            }
            binEdges.add(edges);
        }

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "KBinsDiscretizer");
        Validation.checkMatrix(X, -1);
        if (X.cols() != binEdges.size()) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + binEdges.size()
                    + " features, got " + X.cols());
        }

        int n = X.rows(), m = X.cols();

        if (encodeOrdinal) {
            double[][] result = new double[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    result[i][j] = digitize(X.get(i, j), binEdges.get(j));
                }
            }
            return new Matrix(result);
        } else {
            int totalCols = m * nBins;
            double[][] result = new double[n][totalCols];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    int bin = (int) digitize(X.get(i, j), binEdges.get(j));
                    int colOffset = j * nBins;
                    if (bin >= 0 && bin < nBins) {
                        result[i][colOffset + bin] = 1.0;
                    }
                }
            }
            return new Matrix(result);
        }
    }

    private static double digitize(double val, double[] edges) {
        for (int k = 1; k < edges.length - 1; k++) {
            if (val < edges[k]) {
                return k - 1;
            }
        }
        return edges.length - 2;
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException(
            "KBinsDiscretizer does not support inverse_transform");
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_bins", nBins);
        params.put("strategy", strategy);
        params.put("encode_ordinal", encodeOrdinal);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }
}
