package org.sklearn.neighbors;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * K-Nearest Neighbors regressor.
 *
 * <p>Predicts target values as the mean of the k nearest training
 * samples using Euclidean distance.
 *
 * <p>Mirrors {@code sklearn.neighbors.KNeighborsRegressor}.
 *
 * <p>Usage:
 * <pre>{@code
 * KNeighborsRegressor knn = new KNeighborsRegressor(3);
 * knn.fit(X, y);
 * Vector preds = knn.predict(X_test);
 * }</pre>
 */
public class KNeighborsRegressor implements Predictor<Matrix, Vector, Vector> {

    private int k;
    private boolean fitted;
    private Matrix xTrain;
    private Vector yTrain;
    private int nFeatures;

    public KNeighborsRegressor(int k) {
        this.k = k;
    }

    @Override
    public KNeighborsRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.xTrain = new Matrix(X);
        this.yTrain = new Vector(y);
        this.nFeatures = X.cols();
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "KNeighborsRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures);
        }

        int n = X.rows();
        int nTrain = xTrain.rows();
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            double[] distances = new double[nTrain];
            for (int j = 0; j < nTrain; j++) {
                double dist = 0.0;
                for (int f = 0; f < nFeatures; f++) {
                    double diff = X.get(i, f) - xTrain.get(j, f);
                    dist += diff * diff;
                }
                distances[j] = dist;
            }

            Integer[] sorted = new Integer[nTrain];
            for (int j = 0; j < nTrain; j++) {
                sorted[j] = j;
            }
            Arrays.sort(sorted, (a, b) -> Double.compare(distances[a], distances[b]));

            double sum = 0.0;
            int nNeighbors = Math.min(k, nTrain);
            for (int j = 0; j < nNeighbors; j++) {
                sum += yTrain.get(sorted[j]);
            }
            preds[i] = sum / nNeighbors;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "KNeighborsRegressor");
        Vector pred = predict(X);
        double ssRes = 0.0, ssTot = 0.0, yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_neighbors", k);
        return Collections.unmodifiableMap(params);
    }
}
