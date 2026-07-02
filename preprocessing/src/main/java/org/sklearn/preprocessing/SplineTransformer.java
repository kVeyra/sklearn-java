package org.sklearn.preprocessing;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * SplineTransformer generates B-spline basis functions for feature encoding.
 *
 * <p>Mirrors {@code sklearn.preprocessing.SplineTransformer}.
 */
public class SplineTransformer implements Transformer<Matrix, Void> {

    private int nKnots;
    private int degree;
    private boolean fitted;
    private int nFeatures;
    private List<double[]> knots;

    /**
     * Create SplineTransformer.
     *
     * @param nKnots number of knots (including boundaries)
     * @param degree  degree of the spline (1 = linear, 2 = quadratic, 3 = cubic)
     */
    public SplineTransformer(int nKnots, int degree) {
        if (nKnots < 2) throw new IllegalArgumentException("nKnots must be >= 2");
        if (degree < 1) throw new IllegalArgumentException("degree must be >= 1");
        this.nKnots = nKnots;
        this.degree = degree;
    }

    @Override
    public SplineTransformer fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        nFeatures = m;

        knots = new ArrayList<>();
        for (int j = 0; j < m; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X.get(i, j);
            Arrays.sort(col);

            double[] k = new double[nKnots];
            for (int i = 0; i < nKnots; i++) {
                int idx = (int) Math.round((double) i * (n - 1) / (nKnots - 1));
                k[i] = col[Math.min(idx, n - 1)];
            }
            knots.add(k);
        }

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SplineTransformer");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        int n = X.rows();
        int m = X.cols();
        int nBases = nKnots + degree - 1;
        double[][] result = new double[n][m * nBases];

        for (int j = 0; j < m; j++) {
            double[] k = knots.get(j);
            for (int i = 0; i < n; i++) {
                double x = X.get(i, j);
                double[] bases = bsplineBases(x, k, degree);
                for (int b = 0; b < bases.length; b++) {
                    result[i][j * nBases + b] = bases[b];
                }
            }
        }
        return new Matrix(result);
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("inverseTransform not supported for SplineTransformer");
    }

    public boolean isFitted() { return fitted; }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_knots", nKnots);
        params.put("degree", degree);
        return params;
    }

    private double[] bsplineBases(double x, double[] knots, int deg) {
        int n = knots.length;
        int nb = n + deg - 1;
        double[] bases = new double[nb];

        for (int i = 0; i < nb; i++) {
            bases[i] = bsplineBasis(x, i, deg, knots);
        }

        double sum = 0;
        for (double v : bases) sum += v;
        if (sum > 0) {
            for (int i = 0; i < nb; i++) bases[i] /= sum;
        }
        return bases;
    }

    private double bsplineBasis(double x, int i, int deg, double[] knots) {
        if (deg == 0) {
            if (i < knots.length - 1) {
                return (x >= knots[i] && x < knots[i + 1]) ? 1.0 : 0.0;
            }
            return 0;
        }
        double left = 0, right = 0;
        double d1 = (i + deg < knots.length) ? knots[i + deg] - knots[i] : 0;
        double d2 = (i + deg + 1 < knots.length) ? knots[i + deg + 1] - knots[i + 1] : 0;

        if (d1 > 0) {
            left = (x - knots[i]) / d1 * bsplineBasis(x, i, deg - 1, knots);
        }
        if (d2 > 0) {
            right = (knots[i + deg + 1] - x) / d2 * bsplineBasis(x, i + 1, deg - 1, knots);
        }
        return left + right;
    }
}
