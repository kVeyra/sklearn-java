package org.sklearn.neighbors;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Nearest Centroid classifier.
 *
 * <p>Each class is represented by the centroid of its training samples.
 * Prediction is the class with the nearest centroid.
 *
 * <p>Mirrors {@code sklearn.neighbors.NearestCentroid}.
 *
 * <p>Usage:
 * <pre>{@code
 * NearestCentroid nc = new NearestCentroid();
 * nc.fit(X, y);
 * Vector preds = nc.predict(X_test);
 * }</pre>
 */
public class NearestCentroid implements Predictor<Matrix, Vector, Vector> {

    private boolean fitted;
    private Map<Integer, Vector> centroids;
    private int[] classes;
    private int nFeatures;

    @Override
    public NearestCentroid fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();

        Map<Integer, List<double[]>> groups = new LinkedHashMap<>();
        for (int i = 0; i < y.size(); i++) {
            int label = (int) y.get(i);
            groups.computeIfAbsent(label, k -> new ArrayList<>());
            double[] row = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                row[j] = X.get(i, j);
            }
            groups.get(label).add(row);
        }

        classes = groups.keySet().stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        centroids = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<double[]>> entry : groups.entrySet()) {
            List<double[]> samples = entry.getValue();
            double[] centroid = new double[nFeatures];
            for (double[] sample : samples) {
                for (int j = 0; j < nFeatures; j++) {
                    centroid[j] += sample[j];
                }
            }
            for (int j = 0; j < nFeatures; j++) {
                centroid[j] /= samples.size();
            }
            centroids.put(entry.getKey(), new Vector(centroid));
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "NearestCentroid");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }

        double[] preds = new double[X.rows()];
        for (int i = 0; i < X.rows(); i++) {
            double bestDist = Double.MAX_VALUE;
            int bestClass = classes[0];
            for (Map.Entry<Integer, Vector> entry : centroids.entrySet()) {
                double dist = 0;
                for (int j = 0; j < nFeatures; j++) {
                    double diff = X.get(i, j) - entry.getValue().get(j);
                    dist += diff * diff;
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    bestClass = entry.getKey();
                }
            }
            preds[i] = bestClass;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return org.sklearn.metrics.ClassificationMetrics.accuracyScore(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes", classes);
        params.put("centroids", centroids);
        return params;
    }

    public Map<Integer, Vector> getCentroids() {
        return centroids;
    }

    public int[] getClasses() {
        return classes;
    }
}
