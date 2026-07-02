package org.sklearn.semi_supervised;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Self-training classifier for semi-supervised learning.
 *
 * <p>Iteratively trains a base classifier on labeled data, then
 * predicts labels for unlabeled data and adds the most confident
 * predictions to the training set.
 *
 * <p>Mirrors {@code sklearn.semi_supervised.SelfTrainingClassifier}.
 */
public class SelfTrainingClassifier implements Predictor<Matrix, Vector, Vector> {

    private Predictor<Matrix, Vector, Vector> baseEstimator;
    private double threshold;
    private int maxIter;
    private boolean fitted;
    private int nFeatures;
    private int[] classes;
    private Vector labels;

    /**
     * Create SelfTrainingClassifier.
     *
     * @param baseEstimator the supervised classifier to wrap
     * @param threshold     confidence threshold for adding labels
     * @param maxIter       maximum number of iterations
     */
    public SelfTrainingClassifier(Predictor<Matrix, Vector, Vector> baseEstimator,
                                  double threshold, int maxIter) {
        this.baseEstimator = baseEstimator;
        this.threshold = threshold;
        this.maxIter = maxIter;
    }

    @Override
    public SelfTrainingClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();
        int n = X.rows();

        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            if (y.get(i) >= 0) seen.add((int) y.get(i));
        }
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);

        double[] currentLabels = new double[n];
        for (int i = 0; i < n; i++) currentLabels[i] = y.get(i);

        for (int iter = 0; iter < maxIter; iter++) {
            List<Integer> labeledList = new ArrayList<>();
            List<Integer> unlabeledList = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (currentLabels[i] >= 0) {
                    labeledList.add(i);
                } else {
                    unlabeledList.add(i);
                }
            }

            if (unlabeledList.isEmpty()) break;

            int[] lIdx = labeledList.stream().mapToInt(Integer::intValue).toArray();
            Matrix xLabeled = extractRows(X, lIdx);
            Vector yLabeled = extractRows(new Vector(currentLabels), lIdx);

            Predictor<Matrix, Vector, Vector> est = cloneEstimator(baseEstimator);
            est.fit(xLabeled, yLabeled);

            boolean added = false;
            for (int u : unlabeledList) {
                Matrix xRow = new Matrix(1, nFeatures);
                for (int j = 0; j < nFeatures; j++) xRow.set(0, j, X.get(u, j));
                Vector pred = est.predict(xRow);
                double label = pred.get(0);
                double conf = computeConfidence(est, xRow, label);

                if (conf >= threshold) {
                    currentLabels[u] = label;
                    added = true;
                }
            }

            if (!added) break;
        }

        labels = new Vector(currentLabels);
        baseEstimator.fit(X, labels);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "SelfTrainingClassifier");
        return baseEstimator.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return org.sklearn.metrics.ClassificationMetrics.accuracyScore(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("threshold", threshold);
        params.put("classes_", classes);
        return params;
    }

    public Vector getLabels() { return labels; }
    public int[] getClasses() { return classes; }
    public boolean isFitted() { return fitted; }

    @SuppressWarnings("unchecked")
    private Predictor<Matrix, Vector, Vector> cloneEstimator(
            Predictor<Matrix, Vector, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator", e);
        }
    }

    private double computeConfidence(Predictor<Matrix, Vector, Vector> est, Matrix xRow, double predictedLabel) {
        if (est instanceof org.sklearn.multiclass.OneVsRestClassifier) {
            org.sklearn.multiclass.OneVsRestClassifier ovr =
                (org.sklearn.multiclass.OneVsRestClassifier) est;
            int[] classes = ovr.getClasses();
            Matrix decision = ovr.decisionFunction(xRow);
            double maxScore = -Double.MAX_VALUE;
            for (int c = 0; c < classes.length; c++) {
                maxScore = Math.max(maxScore, decision.get(0, c));
            }
            double thisScore = 0;
            for (int c = 0; c < classes.length; c++) {
                if (classes[c] == (int) predictedLabel) {
                    thisScore = decision.get(0, c);
                    break;
                }
            }
            if (maxScore > 0) return thisScore / maxScore;
            return thisScore >= 0.5 ? 1.0 : 0.0;
        }
        return 1.0;
    }

    private static Matrix extractRows(Matrix X, int[] indices) {
        int n = indices.length;
        int m = X.cols();
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) data[i][j] = X.get(indices[i], j);
        }
        return new Matrix(data);
    }

    private static Vector extractRows(Vector y, int[] indices) {
        int n = indices.length;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = y.get(indices[i]);
        return new Vector(data);
    }
}
