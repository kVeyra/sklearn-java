package org.sklearn.cluster;

import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * K-Means clustering.
 *
 * <p>Partitions samples into n_clusters using Lloyd's algorithm.
 * Initialization uses k-means++ for better convergence.
 * Returns cluster labels and centroids, and computes inertia
 * (sum of squared distances to nearest centroid).
 *
 * <p>Mirrors {@code sklearn.cluster.KMeans}.
 *
 * <p>Usage:
 * <pre>{@code
 * KMeans kmeans = new KMeans(3);
 * kmeans.fit(X);
 * int[] labels = kmeans.getLabels();
 * Matrix centers = kmeans.getClusterCenters();
 * }</pre>
 */
public class KMeans {

    private int nClusters;
    private int maxIter;
    private double tol;
    private int nInit;
    private long seed;
    private boolean fitted;
    private Matrix clusterCenters;
    private int[] labels;
    private double inertia;
    private int nFeatures;

    /**
     * Create a KMeans clusterer.
     *
     * @param nClusters number of clusters
     */
    public KMeans(int nClusters) {
        this(nClusters, 300, 1e-4, 10, 42);
    }

    /**
     * Create a KMeans clusterer with full parameters.
     *
     * @param nClusters number of clusters
     * @param maxIter   maximum iterations per run
     * @param tol       relative tolerance for convergence
     * @param nInit     number of runs with different seeds
     * @param seed      random seed
     */
    public KMeans(int nClusters, int maxIter, double tol, int nInit, long seed) {
        this.nClusters = nClusters;
        this.maxIter = maxIter;
        this.tol = tol;
        this.nInit = nInit;
        this.seed = seed;
    }

    public KMeans fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        double bestInertia = Double.MAX_VALUE;
        Matrix bestCenters = null;
        int[] bestLabels = null;

        for (int run = 0; run < nInit; run++) {
            RandomGenerator rng = new RandomGenerator(seed + run * 1000);
            Matrix centers = initKMeansPlusPlus(X, nClusters, rng);

            int[] currentLabels = new int[n];
            double currentInertia = 0.0;
            int iter;

            for (iter = 0; iter < maxIter; iter++) {
                // Assign labels
                currentInertia = 0.0;
                for (int i = 0; i < n; i++) {
                    double minDist = Double.MAX_VALUE;
                    int bestK = 0;
                    for (int k = 0; k < nClusters; k++) {
                        double dist = 0.0;
                        for (int j = 0; j < m; j++) {
                            double diff = X.get(i, j) - centers.get(k, j);
                            dist += diff * diff;
                        }
                        if (dist < minDist) {
                            minDist = dist;
                            bestK = k;
                        }
                    }
                    currentLabels[i] = bestK;
                    currentInertia += minDist;
                }

                // Update centroids
                Matrix newCenters = new Matrix(nClusters, m);
                int[] counts = new int[nClusters];
                for (int i = 0; i < n; i++) {
                    int k = currentLabels[i];
                    counts[k]++;
                    for (int j = 0; j < m; j++) {
                        newCenters.set(k, j, newCenters.get(k, j) + X.get(i, j));
                    }
                }
                for (int k = 0; k < nClusters; k++) {
                    if (counts[k] > 0) {
                        for (int j = 0; j < m; j++) {
                            newCenters.set(k, j, newCenters.get(k, j) / counts[k]);
                        }
                    } else {
                        // Re-initialize empty cluster
                        int ri = rng.nextInt(n);
                        for (int j = 0; j < m; j++) {
                            newCenters.set(k, j, X.get(ri, j));
                        }
                    }
                }

                // Check convergence
                double maxShift = 0.0;
                for (int k = 0; k < nClusters; k++) {
                    double shift = 0.0;
                    for (int j = 0; j < m; j++) {
                        double diff = newCenters.get(k, j) - centers.get(k, j);
                        shift += diff * diff;
                    }
                    maxShift = Math.max(maxShift, Math.sqrt(shift));
                }
                centers = newCenters;

                if (maxShift < tol) {
                    break;
                }

                // Recompute inertia for convergence check
                currentInertia = 0.0;
                for (int i = 0; i < n; i++) {
                    double minDist = Double.MAX_VALUE;
                    for (int k = 0; k < nClusters; k++) {
                        double dist = 0.0;
                        for (int j = 0; j < m; j++) {
                            double diff = X.get(i, j) - centers.get(k, j);
                            dist += diff * diff;
                        }
                        if (dist < minDist) {
                            minDist = dist;
                        }
                    }
                    currentInertia += minDist;
                }
            }

            if (currentInertia < bestInertia) {
                bestInertia = currentInertia;
                bestCenters = centers;
                bestLabels = currentLabels;
            }
        }

        this.clusterCenters = bestCenters;
        this.labels = bestLabels;
        this.inertia = bestInertia;
        this.fitted = true;
        return this;
    }

    /**
     * Predict the closest cluster for each sample.
     */
    public int[] predict(Matrix X) {
        Validation.checkFitted(fitted, "KMeans");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }
        int n = X.rows();
        int m = X.cols();
        int[] preds = new int[n];
        for (int i = 0; i < n; i++) {
            double minDist = Double.MAX_VALUE;
            int bestK = 0;
            for (int k = 0; k < nClusters; k++) {
                double dist = 0.0;
                for (int j = 0; j < m; j++) {
                    double diff = X.get(i, j) - clusterCenters.get(k, j);
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bestK = k;
                }
            }
            preds[i] = bestK;
        }
        return preds;
    }

    /**
     * Score as negative inertia (higher is better).
     */
    public double score(Matrix X) {
        Validation.checkFitted(fitted, "KMeans");
        int[] preds = predict(X);
        int n = X.rows();
        int m = X.cols();
        double totalDist = 0.0;
        for (int i = 0; i < n; i++) {
            int k = preds[i];
            double dist = 0.0;
            for (int j = 0; j < m; j++) {
                double diff = X.get(i, j) - clusterCenters.get(k, j);
                dist += diff * diff;
            }
            totalDist += dist;
        }
        return -totalDist;
    }

    private Matrix initKMeansPlusPlus(Matrix X, int k, RandomGenerator rng) {
        int n = X.rows();
        int m = X.cols();
        double[][] centers = new double[k][m];

        int firstIdx = rng.nextInt(n);
        for (int j = 0; j < m; j++) {
            centers[0][j] = X.get(firstIdx, j);
        }

        double[] minDist = new double[n];
        for (int i = 0; i < n; i++) {
            double dist = 0.0;
            for (int j = 0; j < m; j++) {
                double diff = X.get(i, j) - centers[0][j];
                dist += diff * diff;
            }
            minDist[i] = dist;
        }

        for (int c = 1; c < k; c++) {
            double totalWeight = 0.0;
            for (int i = 0; i < n; i++) {
                totalWeight += minDist[i];
            }

            double r = rng.nextDouble() * totalWeight;
            double cumSum = 0.0;
            int chosen = 0;
            for (int i = 0; i < n; i++) {
                cumSum += minDist[i];
                if (cumSum >= r) {
                    chosen = i;
                    break;
                }
            }

            for (int j = 0; j < m; j++) {
                centers[c][j] = X.get(chosen, j);
            }

            for (int i = 0; i < n; i++) {
                double dist = 0.0;
                for (int j = 0; j < m; j++) {
                    double diff = X.get(i, j) - centers[c][j];
                    dist += diff * diff;
                }
                if (dist < minDist[i]) {
                    minDist[i] = dist;
                }
            }
        }

        return new Matrix(centers);
    }

    public Matrix getClusterCenters() {
        return clusterCenters;
    }

    public int[] getLabels() {
        return labels;
    }

    public double getInertia() {
        return inertia;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_clusters", nClusters);
        params.put("max_iter", maxIter);
        params.put("tol", tol);
        params.put("n_init", nInit);
        return Collections.unmodifiableMap(params);
    }
}
