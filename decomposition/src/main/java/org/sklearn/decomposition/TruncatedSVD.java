package org.sklearn.decomposition;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Dimensionality reduction using truncated SVD (aka LSA).
 *
 * <p>This transformer performs linear dimensionality reduction via
 * truncated singular value decomposition. Unlike PCA, this estimator
 * does not center the data before computing the SVD.
 *
 * <p>Mirrors {@code sklearn.decomposition.TruncatedSVD}.
 *
 * <p>Usage:
 * <pre>{@code
 * TruncatedSVD svd = new TruncatedSVD(2);
 * svd.fit(X, null);
 * Matrix Xreduced = svd.transform(X);
 * }</pre>
 */
public class TruncatedSVD implements Transformer<Matrix, Void> {

    private int nComponents;
    private boolean fitted;
    private int nFeatures;
    private Matrix components;
    private Vector explainedVariance;
    private Vector explainedVarianceRatio;
    private Vector singularValues;

    /**
     * Create TruncatedSVD.
     *
     * @param nComponents number of components to keep
     */
    public TruncatedSVD(int nComponents) {
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        this.nComponents = nComponents;
    }

    @Override
    public TruncatedSVD fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        nFeatures = X.cols();
        int n = X.rows();
        int m = X.cols();
        int k = Math.min(nComponents, Math.min(n, m));

        double[][] a = X.toArray();

        double[][] ata = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int t = 0; t < n; t++) {
                    sum += a[t][i] * a[t][j];
                }
                ata[i][j] = sum;
            }
        }

        Eigensystem es = eigenDecompose(ata, m);
        int[] order = sortDescending(es.values);

        double[][] vt = new double[k][m];
        double[] sv = new double[k];
        for (int i = 0; i < k; i++) {
            int idx = order[i];
            sv[i] = Math.sqrt(Math.max(0, es.values[idx]));
            for (int j = 0; j < m; j++) {
                vt[i][j] = es.vectors[idx][j];
            }
        }

        components = new Matrix(vt);
        singularValues = new Vector(sv);

        double totalVar = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                totalVar += a[i][j] * a[i][j];
            }
        }

        double[] ev = new double[k];
        double[] ratio = new double[k];
        for (int i = 0; i < k; i++) {
            ev[i] = sv[i] * sv[i] / (n - 1);
            ratio[i] = totalVar > 0 ? (sv[i] * sv[i]) / totalVar : 0;
        }
        explainedVariance = new Vector(ev);
        explainedVarianceRatio = new Vector(ratio);

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "TruncatedSVD");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        return X.multiply(components.transpose());
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "TruncatedSVD");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nComponents) {
            throw new IllegalArgumentException("Expected " + nComponents + " components, got " + X.cols());
        }
        return X.multiply(components);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_components", nComponents);
        params.put("components", components);
        params.put("explained_variance", explainedVariance);
        params.put("explained_variance_ratio", explainedVarianceRatio);
        params.put("singular_values", singularValues);
        return Collections.unmodifiableMap(params);
    }

    public Matrix getComponents() {
        return components;
    }

    public Vector getSingularValues() {
        return singularValues;
    }

    public Vector getExplainedVarianceRatio() {
        return explainedVarianceRatio;
    }

    // ---- Private helpers ----

    private static class Eigensystem {
        final double[] values;
        final double[][] vectors;
        Eigensystem(double[] values, double[][] vectors) {
            this.values = values;
            this.vectors = vectors;
        }
    }

    private static Eigensystem eigenDecompose(double[][] a, int n) {
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            v[i][i] = 1.0;
        }
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = a[i][i];
        }

        int maxSweeps = 50;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    off += a[i][j] * a[i][j];
                }
            }
            if (off < 1e-12) {
                break;
            }
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = a[p][q];
                    if (Math.abs(apq) < 1e-14) {
                        continue;
                    }
                    double diff = d[q] - d[p];
                    double tau;
                    if (Math.abs(diff) < 1e-14 && Math.abs(apq) < 1e-14) {
                        tau = 0.0;
                    } else {
                        double theta = diff / (2.0 * apq);
                        double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(1.0 + theta * theta));
                        tau = t;
                    }
                    double c = 1.0 / Math.sqrt(1.0 + tau * tau);
                    double s = tau * c;

                    d[p] -= tau * apq;
                    d[q] += tau * apq;
                    a[p][q] = 0.0;

                    for (int i = 0; i < p; i++) {
                        double api = a[i][p];
                        double aqi = a[i][q];
                        a[i][p] = api * c - aqi * s;
                        a[i][q] = api * s + aqi * c;
                        a[p][i] = a[i][p];
                        a[q][i] = a[i][q];
                    }
                    for (int i = p + 1; i < q; i++) {
                        double api = a[p][i];
                        double aqi = a[i][q];
                        a[p][i] = api * c - aqi * s;
                        a[i][q] = api * s + aqi * c;
                        a[i][p] = a[p][i];
                        a[q][i] = a[i][q];
                    }
                    for (int i = q + 1; i < n; i++) {
                        double api = a[p][i];
                        double aqi = a[q][i];
                        a[p][i] = api * c - aqi * s;
                        a[q][i] = api * s + aqi * c;
                        a[i][p] = a[p][i];
                        a[i][q] = a[q][i];
                    }
                    for (int i = 0; i < n; i++) {
                        double vpi = v[i][p];
                        double vqi = v[i][q];
                        v[i][p] = vpi * c - vqi * s;
                        v[i][q] = vpi * s + vqi * c;
                    }
                }
            }
        }
        return new Eigensystem(d, v);
    }

    private static int[] sortDescending(double[] values) {
        int n = values.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(values[b], values[a]));
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = idx[i];
        }
        return result;
    }
}
