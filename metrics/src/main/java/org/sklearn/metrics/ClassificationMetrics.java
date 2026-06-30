package org.sklearn.metrics;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

/**
 * Classification metrics: accuracy, precision, recall, F1, confusion matrix.
 *
 * <p>Mirrors {@code sklearn.metrics} classification functions.
 */
public final class ClassificationMetrics {

    private ClassificationMetrics() {
    }

    /**
     * Accuracy classification score.
     */
    public static double accuracyScore(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int correct = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            if (Math.abs(yTrue.get(i) - yPred.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / yTrue.size();
    }

    /**
     * Compute confusion matrix.
     * Returns matrix where entry [i,j] is count of samples with
     * true class i predicted as class j.
     */
    public static Matrix confusionMatrix(Vector yTrue, Vector yPred, int nClasses) {
        checkLengths(yTrue, yPred);
        Matrix cm = new Matrix(nClasses, nClasses);
        for (int i = 0; i < yTrue.size(); i++) {
            int t = (int) yTrue.get(i);
            int p = (int) yPred.get(i);
            cm.set(t, p, cm.get(t, p) + 1);
        }
        return cm;
    }

    /**
     * Precision: tp / (tp + fp)
     */
    public static double precisionScore(Vector yTrue, Vector yPred, int positiveClass) {
        int tp = 0, fp = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            int t = (int) yTrue.get(i);
            int p = (int) yPred.get(i);
            if (p == positiveClass) {
                if (t == positiveClass) {
                    tp++;
                } else {
                    fp++;
                }
            }
        }
        return tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
    }

    /**
     * Recall: tp / (tp + fn)
     */
    public static double recallScore(Vector yTrue, Vector yPred, int positiveClass) {
        int tp = 0, fn = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            int t = (int) yTrue.get(i);
            int p = (int) yPred.get(i);
            if (t == positiveClass) {
                if (p == positiveClass) {
                    tp++;
                } else {
                    fn++;
                }
            }
        }
        return tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
    }

    /**
     * F1 score: 2 * precision * recall / (precision + recall)
     */
    public static double f1Score(Vector yTrue, Vector yPred, int positiveClass) {
        double p = precisionScore(yTrue, yPred, positiveClass);
        double r = recallScore(yTrue, yPred, positiveClass);
        return p + r == 0 ? 0.0 : 2 * p * r / (p + r);
    }

    private static void checkLengths(Vector a, Vector b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                "yTrue and yPred must have same length: " + a.size() + " vs " + b.size());
        }
    }
}
