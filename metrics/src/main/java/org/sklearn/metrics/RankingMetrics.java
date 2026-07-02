package org.sklearn.metrics;

import org.sklearn.math.Vector;

import java.util.Arrays;

/**
 * Ranking and threshold metrics: ROC curve, AUC, ROC AUC, PR curve.
 *
 * <p>Mirrors {@code sklearn.metrics} ranking functions.
 */
public final class RankingMetrics {

    private RankingMetrics() {
    }

    /**
     * Compute ROC curve (false positive rates, true positive rates, thresholds).
     */
    public static RocCurve rocCurve(Vector yTrue, Vector yScore) {
        checkLengths(yTrue, yScore);
        int n = yTrue.size();

        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(yScore.get(b), yScore.get(a)));

        double[] tpr = new double[n + 1];
        double[] fpr = new double[n + 1];
        double[] thresholds = new double[n];

        int nPos = 0;
        for (int i = 0; i < n; i++) {
            if (yTrue.get(i) > 0.5) {
                nPos++;
            }
        }
        int nNeg = n - nPos;

        tpr[0] = 0;
        fpr[0] = 0;
        int tp = 0;
        int fp = 0;
        for (int i = 0; i < n; i++) {
            if (yTrue.get(idx[i]) > 0.5) {
                tp++;
            } else {
                fp++;
            }
            tpr[i + 1] = (double) tp / nPos;
            fpr[i + 1] = (double) fp / nNeg;
            thresholds[i] = yScore.get(idx[i]);
        }

        return new RocCurve(fpr, tpr, thresholds);
    }

    /**
     * Compute AUC using the trapezoidal rule.
     */
    public static double auc(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        double sum = 0;
        for (int i = 1; i < n; i++) {
            sum += (y[i] + y[i - 1]) * (x[i] - x[i - 1]) / 2.0;
        }
        return Math.abs(sum);
    }

    /**
     * Compute ROC AUC score.
     */
    public static double rocAucScore(Vector yTrue, Vector yScore) {
        RocCurve curve = rocCurve(yTrue, yScore);
        return auc(curve.fpr, curve.tpr);
    }

    /**
     * Compute precision-recall curve.
     */
    public static PrCurve precisionRecallCurve(Vector yTrue, Vector yScore) {
        RocCurve roc = rocCurve(yTrue, yScore);
        int n = roc.thresholds.length;
        double[] precision = new double[n + 1];
        double[] recall = new double[n + 1];

        int nPos = 0;
        for (int i = 0; i < yTrue.size(); i++) {
            if (yTrue.get(i) > 0.5) {
                nPos++;
            }
        }

        Integer[] idx = new Integer[yTrue.size()];
        for (int i = 0; i < yTrue.size(); i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(yScore.get(b), yScore.get(a)));

        precision[0] = 1.0;
        recall[0] = 0.0;
        int tp = 0;
        int fp = 0;
        for (int i = 0; i < n; i++) {
            if (yTrue.get(idx[i]) > 0.5) {
                tp++;
            } else {
                fp++;
            }
            recall[i + 1] = (double) tp / nPos;
            precision[i + 1] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
        }

        return new PrCurve(precision, recall, roc.thresholds);
    }

    /**
     * Compute average precision (AP) score.
     */
    public static double averagePrecisionScore(Vector yTrue, Vector yScore) {
        PrCurve pr = precisionRecallCurve(yTrue, yScore);
        return auc(pr.recall, pr.precision);
    }

    /**
     * Compute precision at top K.
     */
    public static double topKPrecision(Vector yTrue, Vector yScore, int k) {
        return topKScore(yTrue, yScore, k, true);
    }

    /**
     * Compute recall at top K.
     */
    public static double topKRecall(Vector yTrue, Vector yScore, int k) {
        return topKScore(yTrue, yScore, k, false);
    }

    private static double topKScore(Vector yTrue, Vector yScore, int k, boolean precision) {
        checkLengths(yTrue, yScore);
        int n = yTrue.size();
        k = Math.min(k, n);

        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(yScore.get(b), yScore.get(a)));

        int tp = 0;
        int totalPos = 0;
        for (int i = 0; i < n; i++) {
            if (yTrue.get(i) > 0.5) {
                totalPos++;
            }
        }
        for (int i = 0; i < k; i++) {
            if (yTrue.get(idx[i]) > 0.5) {
                tp++;
            }
        }
        if (precision) {
            return (double) tp / k;
        }
        return totalPos > 0 ? (double) tp / totalPos : 0;
    }

    /**
     * Compute Discounted Cumulative Gain (DCG).
     */
    public static double dcgScore(Vector yTrue, Vector yScore) {
        checkLengths(yTrue, yScore);
        int n = yTrue.size();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(yScore.get(b), yScore.get(a)));

        double dcg = 0;
        for (int i = 0; i < n; i++) {
            double rel = yTrue.get(idx[i]);
            dcg += (Math.pow(2, rel) - 1) / (Math.log(i + 2) / Math.log(2));
        }
        return dcg;
    }

    /**
     * Compute Normalized Discounted Cumulative Gain (NDCG).
     */
    public static double ndcgScore(Vector yTrue, Vector yScore) {
        double dcg = dcgScore(yTrue, yScore);
        double[] ideal = new double[yTrue.size()];
        for (int i = 0; i < yTrue.size(); i++) {
            ideal[i] = yTrue.get(i);
        }
        Arrays.sort(ideal);
        for (int i = 0; i < ideal.length / 2; i++) {
            double tmp = ideal[i];
            ideal[i] = ideal[ideal.length - 1 - i];
            ideal[ideal.length - 1 - i] = tmp;
        }
        Vector idealTrue = new Vector(ideal);
        Vector idealScore = new Vector(ideal);
        double idcg = dcgScore(idealTrue, idealScore);
        return idcg > 0 ? dcg / idcg : 0;
    }

    /**
     * ROC curve result.
     */
    public static class RocCurve {
        public final double[] fpr;
        public final double[] tpr;
        public final double[] thresholds;

        RocCurve(double[] fpr, double[] tpr, double[] thresholds) {
            this.fpr = fpr;
            this.tpr = tpr;
            this.thresholds = thresholds;
        }
    }

    /**
     * Precision-recall curve result.
     */
    public static class PrCurve {
        public final double[] precision;
        public final double[] recall;
        public final double[] thresholds;

        PrCurve(double[] precision, double[] recall, double[] thresholds) {
            this.precision = precision;
            this.recall = recall;
            this.thresholds = thresholds;
        }
    }

    private static void checkLengths(Vector a, Vector b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                "yTrue and yScore must have same length: " + a.size() + " vs " + b.size());
        }
    }
}
