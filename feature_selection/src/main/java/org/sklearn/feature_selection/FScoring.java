package org.sklearn.feature_selection;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import java.util.*;

public final class FScoring {

    private FScoring() {}

    public static FTestResult fClassif(Matrix X, Vector y) {
        int n = X.rows();
        int m = X.cols();
        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) uniqueLabels.add((int) y.get(i));
        int[] classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        int nClasses = classes.length;

        double[] globalMean = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += X.get(i, j);
            globalMean[j] = sum / n;
        }

        double[][] classMeans = new double[nClasses][m];
        int[] classCounts = new int[nClasses];
        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            classCounts[c]++;
            for (int j = 0; j < m; j++) classMeans[c][j] += X.get(i, j);
        }
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < m; j++) classMeans[c][j] /= classCounts[c];
        }

        double[] scores = new double[m];
        double[] pvalues = new double[m];
        for (int j = 0; j < m; j++) {
            double ssBetween = 0, ssWithin = 0;
            for (int c = 0; c < nClasses; c++) {
                double diff = classMeans[c][j] - globalMean[j];
                ssBetween += classCounts[c] * diff * diff;
            }
            for (int i = 0; i < n; i++) {
                int c = indexOf(classes, (int) y.get(i));
                double diff = X.get(i, j) - classMeans[c][j];
                ssWithin += diff * diff;
            }
            double fStat = nClasses > 1 && ssWithin > 0
                ? (ssBetween / (nClasses - 1)) / (ssWithin / (n - nClasses))
                : 0;
            scores[j] = fStat;
            pvalues[j] = fStat > 0 ? fDistributionPValue(fStat, nClasses - 1, n - nClasses) : 1.0;
        }
        return new FTestResult(scores, pvalues);
    }

    public static FTestResult fRegression(Matrix X, Vector y) {
        int n = X.rows();
        int m = X.cols();
        double yMean = y.mean();
        double[] scores = new double[m];
        double[] pvalues = new double[m];

        for (int j = 0; j < m; j++) {
            double xSum = 0;
            for (int i = 0; i < n; i++) xSum += X.get(i, j);
            double xMean = xSum / n;

            double ssReg = 0, ssRes = 0, ssXX = 0;
            for (int i = 0; i < n; i++) {
                double xDiff = X.get(i, j) - xMean;
                double yDiff = y.get(i) - yMean;
                ssReg += xDiff * yDiff;
                ssRes += yDiff * yDiff;
                ssXX += xDiff * xDiff;
            }
            double numer = ssReg * ssReg;
            double fStat = ssXX > 0 && ssRes > 0
                ? (numer / ssXX) / (ssRes / (n - 2))
                : 0;
            scores[j] = fStat;
            pvalues[j] = fStat > 0 ? fDistributionPValue(fStat, 1, n - 2) : 1.0;
        }
        return new FTestResult(scores, pvalues);
    }

    public static double[] rRegression(Matrix X, Vector y) {
        int n = X.rows();
        int m = X.cols();
        double yMean = y.mean();
        double[] r = new double[m];
        for (int j = 0; j < m; j++) {
            double xSum = 0;
            for (int i = 0; i < n; i++) xSum += X.get(i, j);
            double xMean = xSum / n;
            double num = 0, denX = 0, denY = 0;
            for (int i = 0; i < n; i++) {
                double xd = X.get(i, j) - xMean;
                double yd = y.get(i) - yMean;
                num += xd * yd;
                denX += xd * xd;
                denY += yd * yd;
            }
            r[j] = denX > 0 && denY > 0 ? num / Math.sqrt(denX * denY) : 0;
        }
        return r;
    }

    public static class FTestResult {
        public final double[] statistic;
        public final double[] pvalue;
        FTestResult(double[] statistic, double[] pvalue) {
            this.statistic = statistic;
            this.pvalue = pvalue;
        }
    }

    private static int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) if (arr[i] == val) return i;
        return 0;
    }

    static double fDistributionPValue(double f, double df1, double df2) {
        if (df2 <= 0 || f < 0) return 1.0;
        double x = df1 * f / (df1 * f + df2);
        return 1.0 - regularizedIncompleteBeta(df1 / 2.0, df2 / 2.0, x);
    }

    static double regularizedIncompleteGamma(double a, double x) {
        if (x < 0 || a <= 0) return 0;
        if (x < a + 1) return seriesGamma(a, x);
        return 1.0 - continuedFractionGamma(a, x);
    }

    private static double seriesGamma(double a, double x) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 1; n <= 200; n++) {
            ap += 1;
            del *= x / ap;
            sum += del;
            if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
        }
        return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
    }

    private static double continuedFractionGamma(double a, double x) {
        double b = x + 1 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= 200; i++) {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-15) break;
        }
        return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
    }

    static double regularizedIncompleteBeta(double a, double b, double x) {
        if (x < 0 || x > 1) return 0;
        if (x == 0 || x == 1) return x == 0 ? 0 : 1;
        double bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b)
            + a * Math.log(x) + b * Math.log(1 - x));
        if (x < (a + 1) / (a + b + 2)) return bt * betaCF(a, b, x) / a;
        return 1 - bt * betaCF(b, a, 1 - x) / b;
    }

    private static double betaCF(double a, double b, double x) {
        double qab = a + b, qap = a + 1, qam = a - 1;
        double c = 1.0, d = 1.0 - qab * x / qap;
        if (Math.abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 200; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-15) break;
        }
        return h;
    }

    private static double lgamma(double x) {
        double[] coef = {76.18009172947146, -86.50532032941677,
            24.01409824083091, -1.231739572450155,
            0.1208650973866179e-2, -0.5395239384953e-5};
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) {
            y += 1.0;
            ser += coef[j] / y;
        }
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
}
