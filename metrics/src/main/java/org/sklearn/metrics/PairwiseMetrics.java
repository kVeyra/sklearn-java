package org.sklearn.metrics;

import org.sklearn.math.Matrix;

/**
 * Pairwise distance and kernel metrics.
 *
 * <p>Mirrors {@code sklearn.metrics.pairwise} functions.
 */
public final class PairwiseMetrics {

    private PairwiseMetrics() {
    }

    /**
     * Compute Euclidean distance matrix between rows of X.
     */
    public static Matrix euclideanDistances(Matrix X) {
        return euclideanDistances(X, X);
    }

    /**
     * Compute Euclidean distance matrix between rows of X and Y.
     */
    public static Matrix euclideanDistances(Matrix X, Matrix Y) {
        int n = X.rows(), m = Y.rows();
        double[][] dist = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < X.cols(); k++) {
                    double diff = X.get(i, k) - Y.get(j, k);
                    sum += diff * diff;
                }
                dist[i][j] = Math.sqrt(sum);
            }
        }
        return new Matrix(dist);
    }

    /**
     * Compute pairwise distances between all rows of X and Y.
     */
    public static Matrix pairwiseDistances(Matrix X, Matrix Y) {
        return euclideanDistances(X, Y);
    }

    /**
     * Compute RBF kernel (similarity) matrix.
     */
    public static Matrix rbfKernel(Matrix X, Matrix Y, double gamma) {
        int n = X.rows(), m = Y.rows();
        double[][] K = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < X.cols(); k++) {
                    double diff = X.get(i, k) - Y.get(j, k);
                    sum += diff * diff;
                }
                K[i][j] = Math.exp(-gamma * sum);
            }
        }
        return new Matrix(K);
    }

    /**
     * Compute linear kernel matrix.
     */
    public static Matrix linearKernel(Matrix X, Matrix Y) {
        return X.multiply(Y.transpose());
    }

    /**
     * Compute polynomial kernel matrix.
     */
    public static Matrix polynomialKernel(Matrix X, Matrix Y, double degree, double gamma, double coef0) {
        Matrix K = X.multiply(Y.transpose());
        int n = K.rows(), m = K.cols();
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = Math.pow(gamma * K.get(i, j) + coef0, degree);
            }
        }
        return new Matrix(result);
    }

    /**
     * Compute pairwise kernel matrix using the given kernel type.
     */
    public static Matrix pairwiseKernels(Matrix X, Matrix Y, String kernel, double gamma, double degree, double coef0) {
        return switch (kernel) {
            case "linear" -> linearKernel(X, Y);
            case "rbf" -> rbfKernel(X, Y, gamma);
            case "poly" -> polynomialKernel(X, Y, degree, gamma, coef0);
            default -> throw new IllegalArgumentException("Unsupported kernel: " + kernel);
        };
    }

    /**
     * Compute Manhattan (L1) distance matrix between rows of X.
     */
    public static Matrix manhattanDistances(Matrix X) {
        return manhattanDistances(X, X);
    }

    /**
     * Compute Manhattan (L1) distance matrix between rows of X and Y.
     */
    public static Matrix manhattanDistances(Matrix X, Matrix Y) {
        int n = X.rows(), m = Y.rows();
        double[][] dist = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < X.cols(); k++) {
                    sum += Math.abs(X.get(i, k) - Y.get(j, k));
                }
                dist[i][j] = sum;
            }
        }
        return new Matrix(dist);
    }

    /**
     * Compute cosine similarity matrix between rows of X.
     */
    public static Matrix cosineSimilarity(Matrix X) {
        return cosineSimilarity(X, X);
    }

    /**
     * Compute cosine similarity matrix between rows of X and Y.
     */
    public static Matrix cosineSimilarity(Matrix X, Matrix Y) {
        int n = X.rows(), m = Y.rows(), d = X.cols();
        double[][] sim = new double[n][m];

        double[] normX = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < d; k++) {
                sum += X.get(i, k) * X.get(i, k);
            }
            normX[i] = Math.sqrt(sum);
        }
        double[] normY = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0;
            for (int k = 0; k < d; k++) {
                sum += Y.get(j, k) * Y.get(j, k);
            }
            normY[j] = Math.sqrt(sum);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double dot = 0;
                for (int k = 0; k < d; k++) {
                    dot += X.get(i, k) * Y.get(j, k);
                }
                double den = normX[i] * normY[j];
                sim[i][j] = den > 0 ? dot / den : 0;
            }
        }
        return new Matrix(sim);
    }

    /**
     * Compute sigmoid kernel matrix between rows of X.
     */
    public static Matrix sigmoidKernel(Matrix X, double gamma, double coef0) {
        return sigmoidKernel(X, X, gamma, coef0);
    }

    /**
     * Compute sigmoid kernel matrix between rows of X and Y.
     */
    public static Matrix sigmoidKernel(Matrix X, Matrix Y, double gamma, double coef0) {
        int n = X.rows(), m = Y.rows();
        double[][] K = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double dot = 0;
                for (int k = 0; k < X.cols(); k++) {
                    dot += X.get(i, k) * Y.get(j, k);
                }
                K[i][j] = Math.tanh(gamma * dot + coef0);
            }
        }
        return new Matrix(K);
    }

    /**
     * Compute nan-tolerant Euclidean distances.
     */
    public static Matrix nanEuclideanDistances(Matrix X, Matrix Y) {
        int n = X.rows(), m = Y.rows(), d = X.cols();
        double[][] dist = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                int count = 0;
                for (int k = 0; k < d; k++) {
                    double v1 = X.get(i, k);
                    double v2 = Y.get(j, k);
                    if (Double.isNaN(v1) || Double.isNaN(v2)) {
                        continue;
                    }
                    double diff = v1 - v2;
                    sum += diff * diff;
                    count++;
                }
                dist[i][j] = count > 0 ? Math.sqrt(sum * d / count) : Double.NaN;
            }
        }
        return new Matrix(dist);
    }
}
