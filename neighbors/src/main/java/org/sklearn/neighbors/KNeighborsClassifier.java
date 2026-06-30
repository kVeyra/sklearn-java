package org.sklearn.neighbors;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * K-Nearest Neighbors classifier.
 *
 * <p>Predicts class labels based on majority vote of the k nearest
 * training samples using Euclidean distance.
 *
 * <p>Mirrors {@code sklearn.neighbors.KNeighborsClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * KNeighborsClassifier knn = new KNeighborsClassifier(3);
 * knn.fit(X, y);
 * Vector preds = knn.predict(X_test);
 * }</pre>
 */
public class KNeighborsClassifier implements Predictor<Matrix, Vector, Vector> {

    private int k;
    private boolean fitted;
    private Matrix xTrain;
    private Vector yTrain;
    private int[] classes;
    private int nFeatures;

    public KNeighborsClassifier(int k) {
        this.k = k;
    }

    @Override
    public KNeighborsClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.xTrain = new Matrix(X);
        this.yTrain = new Vector(y);
        this.nFeatures = X.cols();

        Set<Integer> labels = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            labels.add((int) y.get(i));
        }
        this.classes = labels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "KNeighborsClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
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
            int finalI = i;
            Arrays.sort(sorted, (a, b) -> Double.compare(distances[a], distances[b]));

            Map<Integer, Integer> votes = new HashMap<>();
            for (int j = 0; j < Math.min(k, nTrain); j++) {
                int label = (int) yTrain.get(sorted[j]);
                votes.merge(label, 1, Integer::sum);
            }

            int bestLabel = 0, bestCount = -1;
            for (var entry : votes.entrySet()) {
                if (entry.getValue() > bestCount) {
                    bestCount = entry.getValue();
                    bestLabel = entry.getKey();
                }
            }
            preds[i] = bestLabel;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "KNeighborsClassifier");
        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
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
