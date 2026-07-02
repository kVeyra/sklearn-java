package org.sklearn.decomposition;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Principal Component Analysis (PCA) for dimensionality reduction.
 *
 * <p>PCA computes the principal components from the training data using
 * eigendecomposition of the covariance matrix (via the Jacobi algorithm).
 * It supports both n_samples >= n_features (covariance method) and
 * n_samples < n_features (Gram matrix method).
 *
 * <p>Mirrors {@code sklearn.decomposition.PCA}.
 *
 * <p>Usage:
 * <pre>{@code
 * PCA pca = new PCA(2);
 * pca.fit(X, null);
 * Matrix Xreduced = pca.transform(X);
 * }</pre>
 */
public class PCA implements Transformer<Matrix, Void> {

    private final int nComponents;

    private Matrix components;

    private Vector mean;

    private Vector explainedVariance;

    private Vector explainedVarianceRatio;

    private Vector singularValues;

    private boolean fitted;

    private int nFeatures;

    private int nSamples;

    /**
     * Create a PCA transformer that keeps the specified number of components.
     *
     * @param nComponents number of principal components to keep (&ge; 1)
     */
    public PCA(int nComponents) {
        if (nComponents < 1) {
            throw new IllegalArgumentException(
                "nComponents must be >= 1, got: " + nComponents);
        }
        this.nComponents = nComponents;
    }

    @Override
    public PCA fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        nSamples = X.rows();
        nFeatures = X.cols();

        mean = computeMean(X);
        Matrix Xc = center(X, mean);

        int n = nSamples;
        int m = nFeatures;
        int k = Math.min(nComponents, Math.min(n, m));

        if (n >= m) {
            fitViaCovariance(Xc, n, m, k);
        } else {
            fitViaGramMatrix(Xc, n, m, k);
        }

        fitted = true;
        return this;
    }

    private void fitViaCovariance(Matrix Xc, int n, int m, int k) {
        Matrix cov = Xc.transpose().multiply(Xc).multiply(1.0 / (n - 1));

        JacobiResult result = jacobiEigen(cov);
        double[] eigenvalues = result.eigenvalues;
        Matrix eigenvectors = result.eigenvectors;

        int[] indices = sortIndicesDescending(eigenvalues);
        double[] sortedEigenvalues = new double[m];
        double[][] sortedEigenvectors = new double[m][m];
        for (int i = 0; i < m; i++) {
            sortedEigenvalues[i] = eigenvalues[indices[i]];
            for (int j = 0; j < m; j++) {
                sortedEigenvectors[i][j] = eigenvectors.get(j, indices[i]);
            }
        }

        double[][] compData = new double[k][m];
        double[] ev = new double[k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                compData[i][j] = sortedEigenvectors[i][j];
            }
            ev[i] = sortedEigenvalues[i];
        }
        components = new Matrix(compData);
        explainedVariance = new Vector(ev);

        double totalVar = 0.0;
        for (int i = 0; i < m; i++) {
            totalVar += sortedEigenvalues[i];
        }
        double[] ratio = new double[k];
        double[] sv = new double[k];
        for (int i = 0; i < k; i++) {
            ratio[i] = totalVar > 0 ? sortedEigenvalues[i] / totalVar : 0.0;
            sv[i] = Math.sqrt((n - 1) * Math.max(0, sortedEigenvalues[i]));
        }
        explainedVarianceRatio = new Vector(ratio);
        singularValues = new Vector(sv);
    }

    private void fitViaGramMatrix(Matrix Xc, int n, int m, int k) {
        Matrix gram = Xc.multiply(Xc.transpose()).multiply(1.0 / (n - 1));

        JacobiResult result = jacobiEigen(gram);
        double[] eigenvalues = result.eigenvalues;
        Matrix eigenvectors = result.eigenvectors;

        int[] indices = sortIndicesDescending(eigenvalues);
        double[] sortedEigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            sortedEigenvalues[i] = eigenvalues[indices[i]];
        }

        double[][] compData = new double[k][m];
        double[] ev = new double[k];
        double[] sv = new double[k];
        for (int i = 0; i < k; i++) {
            double lambda = Math.max(0, sortedEigenvalues[i]);
            ev[i] = lambda;
            sv[i] = Math.sqrt((n - 1) * lambda);

            Vector vi = new Vector(n);
            for (int j = 0; j < n; j++) {
                vi.set(j, eigenvectors.get(j, indices[i]));
            }

            double scale = 1.0 / Math.sqrt(Math.max(1e-15, (n - 1) * lambda));
            for (int f = 0; f < m; f++) {
                double sum = 0.0;
                for (int s = 0; s < n; s++) {
                    sum += Xc.get(s, f) * vi.get(s);
                }
                compData[i][f] = sum * scale;
            }
        }
        components = new Matrix(compData);
        explainedVariance = new Vector(ev);

        double totalVar = 0.0;
        for (int i = 0; i < n; i++) {
            totalVar += sortedEigenvalues[i];
        }
        double[] ratio = new double[k];
        for (int i = 0; i < k; i++) {
            ratio[i] = totalVar > 0 ? sortedEigenvalues[i] / totalVar : 0.0;
        }
        explainedVarianceRatio = new Vector(ratio);
        singularValues = new Vector(sv);
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "PCA");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }
        Matrix Xc = center(X, mean);
        return Xc.multiply(components.transpose());
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "PCA");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nComponents) {
            throw new IllegalArgumentException(
                "Expected " + nComponents + " components, got " + X.cols());
        }
        Matrix Xproj = X.multiply(components);
        int n = Xproj.rows();
        int m = Xproj.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = Xproj.get(i, j) + mean.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_components", nComponents);
        params.put("components", components);
        params.put("explained_variance", explainedVariance);
        params.put("explained_variance_ratio", explainedVarianceRatio);
        params.put("singular_values", singularValues);
        params.put("mean", mean);
        return Collections.unmodifiableMap(params);
    }

    public boolean isFitted() {
        return fitted;
    }

    public int getNComponents() {
        return nComponents;
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    private static Vector computeMean(Matrix X) {
        int n = X.rows();
        int m = X.cols();
        double[] means = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X.get(i, j);
            }
            means[j] = sum / n;
        }
        return new Vector(means);
    }

    private static Matrix center(Matrix X, Vector mean) {
        int n = X.rows();
        int m = X.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = X.get(i, j) - mean.get(j);
            }
        }
        return new Matrix(result);
    }

    private static int[] sortIndicesDescending(double[] values) {
        int n = values.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Double.compare(values[b], values[a]));
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = indices[i];
        }
        return result;
    }

    private static JacobiResult jacobiEigen(Matrix A) {
        int n = A.rows();
        double[][] a = A.toArray();
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
            double sumOffDiag = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sumOffDiag += Math.abs(a[i][j]);
                }
            }
            if (sumOffDiag < 1e-12) {
                break;
            }

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double maxOff = Math.abs(a[p][q]);
                    if (maxOff < 1e-14) {
                        continue;
                    }

                    double diff = d[q] - d[p];
                    double t;
                    if (Math.abs(diff) < 1e-14 && Math.abs(a[p][q]) < 1e-14) {
                        t = 0.0;
                    } else {
                        double theta = diff / (2.0 * a[p][q]);
                        if (theta >= 0) {
                            t = 1.0 / (theta + Math.sqrt(1.0 + theta * theta));
                        } else {
                            t = -1.0 / (-theta + Math.sqrt(1.0 + theta * theta));
                        }
                    }

                    double c = 1.0 / Math.sqrt(1.0 + t * t);
                    double s = t * c;

                    d[p] -= t * a[p][q];
                    d[q] += t * a[p][q];
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

        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = d[i];
        }
        Matrix eigenvectors = new Matrix(v);

        return new JacobiResult(eigenvalues, eigenvectors);
    }

    private static class JacobiResult {
        final double[] eigenvalues;
        final Matrix eigenvectors;

        JacobiResult(double[] eigenvalues, Matrix eigenvectors) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
        }
    }
}
