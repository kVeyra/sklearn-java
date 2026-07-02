package org.sklearn.metrics;

import org.sklearn.math.Vector;

/**
 * Regression metrics: MAE, MSE, RMSE, RMSLE, R², MAPE, MedAE, MaxError, EVar.
 *
 * <p>Mirrors {@code sklearn.metrics} regression functions.
 */
public final class RegressionMetrics {

    private RegressionMetrics() {
    }

    public static double meanAbsoluteError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.abs(yTrue.get(i) - yPred.get(i));
        }
        return sum / n;
    }

    public static double meanSquaredError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double diff = yTrue.get(i) - yPred.get(i);
            sum += diff * diff;
        }
        return sum / n;
    }

    public static double rootMeanSquaredError(Vector yTrue, Vector yPred) {
        return Math.sqrt(meanSquaredError(yTrue, yPred));
    }

    public static double r2Score(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double yMean = yTrue.mean();
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++) {
            double diff = yTrue.get(i) - yPred.get(i);
            ssRes += diff * diff;
            double diffMean = yTrue.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public static double meanAbsolutePercentageError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double trueVal = Math.abs(yTrue.get(i));
            if (trueVal > 1e-15) {
                sum += Math.abs((yTrue.get(i) - yPred.get(i)) / trueVal);
            }
        }
        return 100.0 * sum / n;
    }

    public static double medianAbsoluteError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double[] absErrors = new double[n];
        for (int i = 0; i < n; i++) {
            absErrors[i] = Math.abs(yTrue.get(i) - yPred.get(i));
        }
        java.util.Arrays.sort(absErrors);
        int mid = n / 2;
        if (n % 2 == 0) {
            return (absErrors[mid - 1] + absErrors[mid]) / 2.0;
        }
        return absErrors[mid];
    }

    public static double maxError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double max = 0;
        for (int i = 0; i < n; i++) {
            double err = Math.abs(yTrue.get(i) - yPred.get(i));
            if (err > max) {
                max = err;
            }
        }
        return max;
    }

    public static double explainedVarianceScore(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double yMean = yTrue.mean();
        double yPredMean = yPred.mean();
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) {
            num += (yTrue.get(i) - yMean) * (yPred.get(i) - yPredMean);
            den += (yTrue.get(i) - yMean) * (yTrue.get(i) - yMean);
        }
        if (den == 0) {
            return 1.0;
        }
        double r = num / den;
        return Math.min(1.0, Math.max(-1.0, r));
    }

    public static double meanSquaredLogError(Vector yTrue, Vector yPred) {
        checkLengths(yTrue, yPred);
        int n = yTrue.size();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double logDiff = Math.log1p(yTrue.get(i)) - Math.log1p(yPred.get(i));
            sum += logDiff * logDiff;
        }
        return sum / n;
    }

    public static double rootMeanSquaredLogError(Vector yTrue, Vector yPred) {
        return Math.sqrt(meanSquaredLogError(yTrue, yPred));
    }

    private static void checkLengths(Vector a, Vector b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                "yTrue and yPred must have same length: " + a.size() + " vs " + b.size());
        }
    }
}
