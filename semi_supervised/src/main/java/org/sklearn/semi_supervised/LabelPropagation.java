package org.sklearn.semi_supervised;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Label Propagation for semi-supervised learning.
 *
 * <p>Propagates labels through the dataset using a graph-based
 * approach where label information is diffused from labeled
 * to unlabeled points.
 *
 * <p>Mirrors {@code sklearn.semi_supervised.LabelPropagation}.
 */
public class LabelPropagation implements Predictor<Matrix, Vector, Vector> {

    private int maxIter;
    private double tol;
    private double gamma;
    private boolean fitted;
    private int nFeatures;
    private int[] classes;
    private Matrix labelDistributions;
    private Matrix trainX;

    /**
     * Create LabelPropagation with default params.
     */
    public LabelPropagation() {
        this(100, 1e-3, 20.0);
    }

    /**
     * Create LabelPropagation.
     */
    public LabelPropagation(int maxIter, double tol, double gamma) {
        this.maxIter = maxIter;
        this.tol = tol;
        this.gamma = gamma;
    }

    @Override
    public LabelPropagation fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();
        int n = X.rows();

        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) seen.add((int) y.get(i));
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);
        int nClasses = classes.length;

        // Build RBF graph
        double[][] W = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dist = 0;
                for (int k = 0; k < nFeatures; k++) {
                    double diff = X.get(i, k) - X.get(j, k);
                    dist += diff * diff;
                }
                W[i][j] = Math.exp(-gamma * dist);
            }
        }

        // Row normalize
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += W[i][j];
            if (sum > 0) {
                for (int j = 0; j < n; j++) W[i][j] /= sum;
            }
        }

        // Initialize soft labels
        double[][] soft = new double[n][nClasses];
        Map<Integer, Integer> labelToIdx = new HashMap<>();
        for (int c = 0; c < nClasses; c++) labelToIdx.put(classes[c], c);

        boolean[] isLabeled = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (y.get(i) >= 0) {
                int idx = labelToIdx.get((int) y.get(i));
                soft[i][idx] = 1.0;
                isLabeled[i] = true;
            } else {
                for (int c = 0; c < nClasses; c++) soft[i][c] = 1.0 / nClasses;
            }
        }

        // Iterate
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] newSoft = new double[n][nClasses];
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < nClasses; c++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        sum += W[i][j] * soft[j][c];
                    }
                    newSoft[i][c] = isLabeled[i] ? soft[i][c] : sum;
                }
            }

            double diff = 0;
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < nClasses; c++) {
                    diff += Math.abs(newSoft[i][c] - soft[i][c]);
                }
            }
            soft = newSoft;
            if (diff < tol * n) break;
        }

        double[][] distData = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int c = 0; c < nClasses; c++) sum += soft[i][c];
            if (sum > 0) {
                for (int c = 0; c < nClasses; c++) distData[i][c] = soft[i][c] / sum;
            } else {
                for (int c = 0; c < nClasses; c++) distData[i][c] = 1.0 / nClasses;
            }
        }
        labelDistributions = new Matrix(distData);
        trainX = new Matrix(X);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "LabelPropagation");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }
        int n = X.rows();
        int nTrain = trainX.rows();
        int nClasses = classes.length;
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            double[] weights = new double[nTrain];
            double wSum = 0;
            for (int j = 0; j < nTrain; j++) {
                double dist = 0;
                for (int k = 0; k < nFeatures; k++) {
                    double diff = X.get(i, k) - trainX.get(j, k);
                    dist += diff * diff;
                }
                weights[j] = Math.exp(-gamma * dist);
                wSum += weights[j];
            }

            double[] classScores = new double[nClasses];
            if (wSum > 0) {
                for (int j = 0; j < nTrain; j++) {
                    for (int c = 0; c < nClasses; c++) {
                        classScores[c] += weights[j] * labelDistributions.get(j, c);
                    }
                }
                for (int c = 0; c < nClasses; c++) classScores[c] /= wSum;
            }

            int bestIdx = 0;
            for (int c = 1; c < nClasses; c++) {
                if (classScores[c] > classScores[bestIdx]) bestIdx = c;
            }
            preds[i] = classes[bestIdx];
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
        params.put("gamma", gamma);
        params.put("classes_", classes);
        params.put("label_distributions_", labelDistributions);
        return params;
    }

    public Matrix getLabelDistributions() { return labelDistributions; }
    public int[] getClasses() { return classes; }
    public boolean isFitted() { return fitted; }
}
