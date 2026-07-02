package org.sklearn.neighbors;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Local Outlier Factor (LOF) for unsupervised outlier detection.
 */
public class LocalOutlierFactor {

    private int nNeighbors;
    private boolean fitted;
    private Matrix xTrain;
    private int nFeatures;
    private double[] negativeOutlierFactor;
    private int[] nNeighborsFound;
    private double[][] lrd;
    private double[][] localReachDensity;
    private double[][] reachDist;

    public LocalOutlierFactor() {
        this(20);
    }

    public LocalOutlierFactor(int nNeighbors) {
        this.nNeighbors = nNeighbors;
    }

    public LocalOutlierFactor fit(Matrix X) {
        Validation.checkMatrix(X, -1);
        this.xTrain = new Matrix(X);
        this.nFeatures = X.cols();
        this.fitted = true;

        int n = X.rows();
        int actualK = Math.min(nNeighbors, n);

        double[][] distances = computeAllDistances(X);
        int[][] nearestIndices = new int[n][actualK];
        double[][] nearestDists = new double[n][actualK];

        for (int i = 0; i < n; i++) {
            Integer[] sorted = sortedIndices(distances[i]);
            for (int j = 0; j < actualK; j++) {
                nearestIndices[i][j] = sorted[j + 1];
                nearestDists[i][j] = distances[i][sorted[j + 1]];
            }
        }

        double[][] reachDistLocal = new double[n][actualK];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < actualK; j++) {
                int ni = nearestIndices[i][j];
                reachDistLocal[i][j] = Math.max(nearestDists[i][j], nearestDists[ni][actualK - 1]);
            }
        }

        double[] lrdLocal = new double[n];
        for (int i = 0; i < n; i++) {
            double sumReach = 0.0;
            for (int j = 0; j < actualK; j++) {
                sumReach += reachDistLocal[i][j];
            }
            lrdLocal[i] = actualK / Math.max(sumReach, 1e-10);
        }

        negativeOutlierFactor = new double[n];
        for (int i = 0; i < n; i++) {
            double sumLrd = 0.0;
            for (int j = 0; j < actualK; j++) {
                sumLrd += lrdLocal[nearestIndices[i][j]];
            }
            double avgLrd = sumLrd / actualK;
            negativeOutlierFactor[i] = -(avgLrd / Math.max(lrdLocal[i], 1e-10));
        }

        this.nNeighborsFound = new int[n];
        Arrays.fill(this.nNeighborsFound, actualK);

        return this;
    }

    public Vector fitPredict(Matrix X) {
        fit(X);
        return decisionFunction(X);
    }

    public Vector decisionFunction(Matrix X) {
        Validation.checkFitted(fitted, "LocalOutlierFactor");
        if (X.rows() != xTrain.rows() || X.cols() != xTrain.cols()) {
            throw new IllegalArgumentException(
                "decisionFunction requires same data as fit. Use fitPredict instead.");
        }
        return new Vector(negativeOutlierFactor);
    }

    public double[] getNegativeOutlierFactor() {
        return negativeOutlierFactor;
    }

    public int[] getNNeighborsFound() {
        return nNeighborsFound;
    }

    private double[][] computeAllDistances(Matrix X) {
        int n = X.rows();
        double[][] dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            dist[i][i] = Double.POSITIVE_INFINITY;
            for (int j = i + 1; j < n; j++) {
                double d = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - X.get(j, f);
                    d += diff * diff;
                }
                dist[i][j] = Math.sqrt(d);
                dist[j][i] = dist[i][j];
            }
        }
        return dist;
    }

    private Integer[] sortedIndices(double[] arr) {
        Integer[] idx = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(arr[a], arr[b]));
        return idx;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_neighbors", nNeighbors);
        return Collections.unmodifiableMap(p);
    }
}
