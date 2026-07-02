package org.sklearn.decomposition;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * FastICA for Independent Component Analysis.
 *
 * <p>Estimates independent source signals from observed mixed signals
 * using the FastICA algorithm with deflationary orthogonalization.
 *
 * <p>Mirrors {@code sklearn.decomposition.FastICA}.
 *
 * <p>Usage:
 * <pre>{@code
 * FastICA ica = new FastICA(2);
 * ica.fit(X);
 * Matrix sources = ica.transform(X);
 * }</pre>
 */
public class FastICA implements Transformer<Matrix, Void> {

    private int nComponents;
    private int maxIter;
    private double tol;
    private boolean fitted;
    private int nFeatures;
    private Matrix mixing;
    private Matrix unmixing;
    private Vector mean;

    /**
     * Create FastICA.
     *
     * @param nComponents number of components to extract
     */
    public FastICA(int nComponents) {
        this(nComponents, 200, 1e-4);
    }

    /**
     * Create FastICA with full control.
     *
     * @param nComponents number of components
     * @param maxIter     maximum iterations
     * @param tol         convergence tolerance
     */
    public FastICA(int nComponents, int maxIter, double tol) {
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        this.nComponents = nComponents;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    @Override
    public FastICA fit(Matrix X, Void y) {
        Validation.checkMatrix(X, -1);
        nFeatures = X.cols();
        int n = X.rows();
        int m = X.cols();
        int k = Math.min(nComponents, Math.min(n, m));

        mean = computeMean(X);
        Matrix Xc = center(X, mean);

        Matrix whitened = whiten(Xc, k, n, m);

        double[][] w = new double[k][k];
        Random rng = new Random(42);

        for (int i = 0; i < k; i++) {
            double[] wi = new double[k];
            for (int j = 0; j < k; j++) {
                wi[j] = rng.nextGaussian() * 0.1;
            }

            for (int iter = 0; iter < maxIter; iter++) {
                double[] wPrev = wi.clone();

                double[] wx = new double[n];
                double[] gwx = new double[n];
                double[] gpwx = new double[n];
                for (int t = 0; t < n; t++) {
                    wx[t] = 0;
                    for (int j = 0; j < k; j++) {
                        wx[t] += wi[j] * whitened.get(t, j);
                    }
                    double val = wx[t];
                    gwx[t] = Math.tanh(val);
                    gpwx[t] = 1 - gwx[t] * gwx[t];
                }

                double[] wNew = new double[k];
                for (int j = 0; j < k; j++) {
                    double sum1 = 0, sum2 = 0;
                    for (int t = 0; t < n; t++) {
                        sum1 += whitened.get(t, j) * gwx[t];
                        sum2 += gpwx[t];
                    }
                    wNew[j] = sum1 / n - sum2 / n * wi[j];
                }

                wi = symmetricOrthogonalize(wNew, i, w);

                double diff = 0;
                double dot = 0;
                for (int j = 0; j < k; j++) {
                    dot += wi[j] * wPrev[j];
                    double d = wi[j] - wPrev[j];
                    diff += d * d;
                }
                if (1 - Math.abs(dot) < tol) {
                    break;
                }
            }

            System.arraycopy(wi, 0, w[i], 0, k);
        }

        unmixing = new Matrix(w);
        Matrix wT = unmixing.transpose();
        mixing = wT.multiply(unmixing).inverse().multiply(wT);

        fitted = true;
        return this;
    }

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "FastICA");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        Matrix Xc = center(X, mean);
        Matrix whitened = whiten(Xc, nComponents, X.rows(), X.cols());
        return whitened.multiply(unmixing.transpose());
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        Validation.checkFitted(fitted, "FastICA");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nComponents) {
            throw new IllegalArgumentException("Expected " + nComponents + " components, got " + X.cols());
        }
        Matrix sources = X.multiply(mixing.transpose());
        int n = sources.rows();
        int m = sources.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = sources.get(i, j) + mean.get(j);
            }
        }
        return new Matrix(result);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_components", nComponents);
        params.put("max_iter", maxIter);
        params.put("tol", tol);
        params.put("mixing", mixing);
        params.put("unmixing", unmixing);
        params.put("mean", mean);
        return Collections.unmodifiableMap(params);
    }

    public Matrix getMixing() {
        return mixing;
    }

    public Matrix getUnmixing() {
        return unmixing;
    }

    // ---- private helpers ----

    private static Vector computeMean(Matrix X) {
        int n = X.rows();
        int m = X.cols();
        double[] means = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0;
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

    private Matrix whiten(Matrix X, int k, int n, int m) {
        Matrix cov = X.transpose().multiply(X).multiply(1.0 / (n - 1));
        int p = Math.min(m, n);

        double[][] a = cov.toArray();
        double[][] v = new double[p][p];
        for (int i = 0; i < p; i++) {
            v[i][i] = 1;
        }
        double[] d = new double[p];
        for (int i = 0; i < p; i++) {
            d[i] = a[i][i];
        }

        int maxSweeps = 50;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0;
            for (int i = 0; i < p; i++) {
                for (int j = i + 1; j < p; j++) {
                    off += a[i][j] * a[i][j];
                }
            }
            if (off < 1e-12) {
                break;
            }

            for (int i = 0; i < p - 1; i++) {
                for (int j = i + 1; j < p; j++) {
                    double bij = a[i][j];
                    if (Math.abs(bij) < 1e-14) {
                        continue;
                    }
                    double tau = (d[j] - d[i]) / (2 * bij);
                    double t = Math.signum(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
                    double c = 1 / Math.sqrt(1 + t * t);
                    double s = t * c;

                    d[i] -= t * bij;
                    d[j] += t * bij;
                    a[i][j] = 0;

                    for (int r = 0; r < i; r++) {
                        double ari = a[r][i], arj = a[r][j];
                        a[r][i] = ari * c - arj * s;
                        a[r][j] = ari * s + arj * c;
                        a[i][r] = a[r][i];
                        a[j][r] = a[r][j];
                    }
                    for (int r = i + 1; r < j; r++) {
                        double air = a[i][r], arj = a[r][j];
                        a[i][r] = air * c - arj * s;
                        a[r][j] = air * s + arj * c;
                        a[r][i] = a[i][r];
                        a[j][r] = a[r][j];
                    }
                    for (int r = j + 1; r < p; r++) {
                        double air = a[i][r], ajr = a[j][r];
                        a[i][r] = air * c - ajr * s;
                        a[j][r] = air * s + ajr * c;
                        a[r][i] = a[i][r];
                        a[r][j] = a[j][r];
                    }
                    for (int r = 0; r < p; r++) {
                        double vri = v[r][i], vrj = v[r][j];
                        v[r][i] = vri * c - vrj * s;
                        v[r][j] = vri * s + vrj * c;
                    }
                }
            }
        }

        double[][] whiteData = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int t = 0; t < p; t++) {
                    sum += X.get(i, t) * v[t][j] / Math.sqrt(Math.max(d[j], 1e-15));
                }
                whiteData[i][j] = sum;
            }
        }
        return new Matrix(whiteData);
    }

    private static double[] symmetricOrthogonalize(double[] wNew, int idx, double[][] w) {
        int k = wNew.length;
        double[] result = wNew.clone();
        for (int i = 0; i <= idx; i++) {
            double dot = 0;
            for (int j = 0; j < k; j++) {
                dot += result[j] * w[i][j];
            }
            for (int j = 0; j < k; j++) {
                result[j] -= dot * w[i][j];
            }
        }
        double norm = 0;
        for (int j = 0; j < k; j++) {
            norm += result[j] * result[j];
        }
        norm = Math.sqrt(norm);
        if (norm > 1e-15) {
            for (int j = 0; j < k; j++) {
                result[j] /= norm;
            }
        }
        return result;
    }
}
