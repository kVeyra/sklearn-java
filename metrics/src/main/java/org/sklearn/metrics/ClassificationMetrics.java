package org.sklearn.metrics;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

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

    /**
     * F-beta score: (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
     */
    public static double fbetaScore(Vector yTrue, Vector yPred, int positiveClass, double beta) {
        double p = precisionScore(yTrue, yPred, positiveClass);
        double r = recallScore(yTrue, yPred, positiveClass);
        double b2 = beta * beta;
        return p + r == 0 ? 0.0 : (1 + b2) * p * r / (b2 * p + r);
    }

    /**
     * Zero-one loss: fraction of misclassified samples.
     */
    public static double zeroOneLoss(Vector yTrue, Vector yPred) {
        return 1.0 - accuracyScore(yTrue, yPred);
    }

    /**
     * Hamming loss: fraction of misclassified labels.
     */
    public static double hammingLoss(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int incorrect = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            if (Math.abs(yTrue.get(i) - yPred.get(i)) >= 0.5) {
                incorrect++;
            }
        }
        return (double) incorrect / yTrue.size();
    }

    /**
     * Jaccard similarity coefficient score.
     */
    public static double jaccardScore(Vector yTrue, Vector yPred, int positiveClass) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            int t = (int) yTrue.get(i);
            int p = (int) yPred.get(i);
            if (p == positiveClass && t == positiveClass) {
                tp++;
            } else if (p == positiveClass && t != positiveClass) {
                fp++;
            } else if (p != positiveClass && t == positiveClass) {
                fn++;
            }
        }
        return tp + fp + fn == 0 ? 0.0 : (double) tp / (tp + fp + fn);
    }

    /**
     * Matthews correlation coefficient.
     */
    public static double matthewsCorrcoef(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            int t = (int) yTrue.get(i);
            int p = (int) yPred.get(i);
            if (t == 1 && p == 1) tp++;
            else if (t == 0 && p == 0) tn++;
            else if (t == 0 && p == 1) fp++;
            else fn++;
        }
        double denom = Math.sqrt((double)(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return denom == 0 ? 0.0 : (tp * tn - fp * fn) / denom;
    }

    /**
     * Cohen's kappa score.
     */
    public static double cohenKappaScore(Vector yTrue, Vector yPred, int nClasses) {
        Matrix cm = confusionMatrix(yTrue, yPred, nClasses);
        int n = yTrue.size();
        double po = 0;
        for (int i = 0; i < nClasses; i++) {
            po += cm.get(i, i);
        }
        po /= n;
        double pe = 0;
        for (int i = 0; i < nClasses; i++) {
            double rowSum = 0, colSum = 0;
            for (int j = 0; j < nClasses; j++) {
                rowSum += cm.get(i, j);
                colSum += cm.get(j, i);
            }
            pe += rowSum * colSum;
        }
        pe /= (n * n);
        return 1 - pe == 0 ? 0.0 : (po - pe) / (1 - pe);
    }

    /**
     * Log loss (logistic loss / cross-entropy loss).
     */
    public static double logLoss(Vector yTrue, Vector yPredProb) {
        checkLengths(yTrue, yPredProb);
        double loss = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, yPredProb.get(i)));
            loss += yTrue.get(i) * Math.log(p) + (1 - yTrue.get(i)) * Math.log(1 - p);
        }
        return -loss / yTrue.size();
    }

    /**
     * Balanced accuracy score.
     */
    public static double balancedAccuracyScore(Vector yTrue, Vector yPred) {
        double tpr = recallScore(yTrue, yPred, 1);
        int tn = 0, nNeg = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            if ((int) yTrue.get(i) == 0) {
                nNeg++;
                if ((int) yPred.get(i) == 0) tn++;
            }
        }
        double tnr = nNeg > 0 ? (double) tn / nNeg : 0;
        return (tpr + tnr) / 2.0;
    }

    /**
     * Top-k accuracy score.
     */
    public static double topKAccuracyScore(Vector yTrue, Matrix yScore, int k) {
        int n = yTrue.size();
        int correct = 0;
        for (int i = 0; i < n; i++) {
            Integer[] idx = new Integer[yScore.cols()];
            for (int j = 0; j < yScore.cols(); j++) {
                idx[j] = j;
            }
            final int row = i;
            java.util.Arrays.sort(idx, (a, b) -> Double.compare(yScore.get(row, b), yScore.get(row, a)));
            int target = (int) yTrue.get(i);
            for (int j = 0; j < Math.min(k, idx.length); j++) {
                if (idx[j] == target) {
                    correct++;
                    break;
                }
            }
        }
        return (double) correct / n;
    }

    /**
     * Build a text classification report.
     */
    public static String classificationReport(Vector yTrue, Vector yPred) {
        Set<Integer> labels = new java.util.LinkedHashSet<>();
        for (int i = 0; i < yTrue.size(); i++) {
            labels.add((int) yTrue.get(i));
            labels.add((int) yPred.get(i));
        }
        int[] classes = labels.stream().mapToInt(Integer::intValue).toArray();
        java.util.Arrays.sort(classes);
        int nClasses = classes.length;
        int last = classes[nClasses - 1];
        Matrix cm = confusionMatrix(yTrue, yPred, last + 1);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-12s %8s %8s %8s %8s%n", "", "precision", "recall", "f1-score", "support"));
        for (int c : classes) {
            int tp = (int) cm.get(c, c);
            int fp = 0, fn = 0;
            for (int j = 0; j <= last; j++) {
                if (j != c) fp += cm.get(j, c);
                if (j != c) fn += cm.get(c, j);
            }
            double p = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
            double r = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            double f1 = p + r == 0 ? 0 : 2 * p * r / (p + r);
            int support = tp + fn;
            sb.append(String.format("%-12s %8.3f %8.3f %8.3f %8d%n", "class " + c, p, r, f1, support));
        }
        sb.append(String.format("%-12s %8s %8s %8s %8s%n", "accuracy", "", "", "",
            String.format("%.3f", accuracyScore(yTrue, yPred))));
        return sb.toString();
    }

    private static void checkLengths(Vector a, Vector b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                "yTrue and yPred must have same length: " + a.size() + " vs " + b.size());
        }
    }
}
