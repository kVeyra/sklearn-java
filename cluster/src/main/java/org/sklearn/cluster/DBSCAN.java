package org.sklearn.cluster;

import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
 *
 * <p>Groups points that are closely packed together, marking points
 * in low-density regions as noise. Core points have at least
 * {@code minPts} neighbors within distance {@code eps}; clusters are
 * formed by connecting core points within {@code eps} of each other.
 *
 * <p>Mirrors {@code sklearn.cluster.DBSCAN} (fit/fit_predict only;
 * DBSCAN has no predict method).
 *
 * <p>Usage:
 * <pre>{@code
 * DBSCAN dbscan = new DBSCAN(0.5, 5);
 * dbscan.fit(X);
 * int[] labels = dbscan.getLabels();
 * }</pre>
 */
public class DBSCAN {

    private double eps;
    private int minPts;
    private boolean fitted;
    private int[] labels;
    private int nFeatures;

    public DBSCAN(double eps, int minPts) {
        this.eps = eps;
        this.minPts = minPts;
    }

    /**
     * Fit DBSCAN and return cluster labels (convenience).
     */
    public int[] fitPredict(Matrix X) {
        fit(X);
        return labels;
    }

    /**
     * Fit the DBSCAN model from training data.
     */
    public DBSCAN fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        labels = new int[n];
        Arrays.fill(labels, -1);

        int clusterId = 0;

        for (int i = 0; i < n; i++) {
            if (labels[i] != -1) {
                continue;
            }

            List<Integer> neighbors = findNeighbors(X, i);

            if (neighbors.size() < minPts) {
                labels[i] = 0;
                continue;
            }

            clusterId++;
            labels[i] = clusterId;

            Queue<Integer> seedSet = new ArrayDeque<>(neighbors);
            while (!seedSet.isEmpty()) {
                int q = seedSet.poll();
                if (labels[q] == 0) {
                    labels[q] = clusterId;
                }
                if (labels[q] != -1) {
                    continue;
                }
                labels[q] = clusterId;

                List<Integer> qNeighbors = findNeighbors(X, q);
                if (qNeighbors.size() >= minPts) {
                    seedSet.addAll(qNeighbors);
                }
            }
        }

        fitted = true;
        return this;
    }

    private List<Integer> findNeighbors(Matrix X, int pointIdx) {
        int n = X.rows();
        int m = X.cols();
        List<Integer> neighbors = new ArrayList<>();
        double epsSq = eps * eps;

        for (int i = 0; i < n; i++) {
            if (i == pointIdx) {
                continue;
            }
            double distSq = 0.0;
            for (int j = 0; j < m; j++) {
                double diff = X.get(pointIdx, j) - X.get(i, j);
                distSq += diff * diff;
            }
            if (distSq <= epsSq) {
                neighbors.add(i);
            }
        }
        return neighbors;
    }

    public int[] getLabels() {
        return labels;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("eps", eps);
        params.put("min_samples", minPts);
        return Collections.unmodifiableMap(params);
    }
}
