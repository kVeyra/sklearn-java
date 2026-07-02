package org.sklearn.neighbors;

import org.sklearn.math.Matrix;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Unsupervised nearest-neighbors for distance queries and index lookups.
 */
public class NearestNeighbors {

    private int nNeighbors;
    private double radius;
    private boolean fitted;
    private Matrix xTrain;
    private int nFeatures;

    public NearestNeighbors() {
        this(5, 1.0);
    }

    public NearestNeighbors(int nNeighbors, double radius) {
        this.nNeighbors = nNeighbors;
        this.radius = radius;
    }

    public NearestNeighbors fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        this.xTrain = new Matrix(X);
        this.nFeatures = X.cols();
        this.fitted = true;
        return this;
    }

    public int[][] kneighbors(Matrix X) {
        return kneighbors(X, nNeighbors);
    }

    public int[][] kneighbors(Matrix X, int k) {
        Validation.checkFitted(fitted, "NearestNeighbors");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        int actualK = Math.min(k, nTrain);
        int[][] indices = new int[n][actualK];

        for (int i = 0; i < n; i++) {
            Integer[] sorted = nearestIndices(X, i, nTrain);
            for (int j = 0; j < actualK; j++) {
                indices[i][j] = sorted[j];
            }
        }
        return indices;
    }

    public Matrix kneighborsDistances(Matrix X) {
        return kneighborsDistances(X, nNeighbors);
    }

    public Matrix kneighborsDistances(Matrix X, int k) {
        Validation.checkFitted(fitted, "NearestNeighbors");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        int actualK = Math.min(k, nTrain);
        Matrix distances = new Matrix(n, actualK);

        for (int i = 0; i < n; i++) {
            Integer[] sorted = nearestIndices(X, i, nTrain);
            for (int j = 0; j < actualK; j++) {
                double d = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - xTrain.get(sorted[j], f);
                    d += diff * diff;
                }
                distances.set(i, j, Math.sqrt(d));
            }
        }
        return distances;
    }

    public List<int[]> radiusNeighbors(Matrix X) {
        return radiusNeighbors(X, radius);
    }

    public List<int[]> radiusNeighbors(Matrix X, double r) {
        Validation.checkFitted(fitted, "NearestNeighbors");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        List<int[]> result = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            List<Integer> neighborList = new ArrayList<>();
            for (int j = 0; j < nTrain; j++) {
                double dist = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - xTrain.get(j, f);
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                if (dist <= r) {
                    neighborList.add(j);
                }
            }
            int[] arr = neighborList.stream().mapToInt(Integer::intValue).toArray();
            result.add(arr);
        }
        return result;
    }

    private Integer[] nearestIndices(Matrix X, int row, int nTrain) {
        double[] distances = new double[nTrain];
        for (int j = 0; j < nTrain; j++) {
            double dist = 0.0;
            for (int f = 0; f < nFeatures; f++) {
                double diff = X.get(row, f) - xTrain.get(j, f);
                dist += diff * diff;
            }
            distances[j] = dist;
        }

        Integer[] sorted = new Integer[nTrain];
        for (int j = 0; j < nTrain; j++) {
            sorted[j] = j;
        }
        Arrays.sort(sorted, (a, b) -> Double.compare(distances[a], distances[b]));
        return sorted;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_neighbors", nNeighbors);
        p.put("radius", radius);
        return Collections.unmodifiableMap(p);
    }
}
