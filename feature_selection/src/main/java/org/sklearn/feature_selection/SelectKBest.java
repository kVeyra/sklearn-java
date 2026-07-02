package org.sklearn.feature_selection;

import org.sklearn.core.Transformer;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Feature selector that selects the k best features based on a scoring function.
 *
 * <p>Mirrors {@code sklearn.feature_selection.SelectKBest}.
 *
 * <p>Usage:
 * <pre>{@code
 * SelectKBest selector = new SelectKBest(5, "f_classif");
 * selector.fit(X, y);
 * Matrix Xreduced = selector.transform(X);
 * }</pre>
 */
public class SelectKBest implements Transformer<Matrix, Matrix> {

    private int k;
    private String scoreFunc;
    private boolean fitted;
    private int[] selectedIndices;
    private double[] scores;
    private double[] pvalues;
    private int nFeatures;

    public SelectKBest() {
        this(10, "f_classif");
    }

    public SelectKBest(int k, String scoreFunc) {
        this.k = k;
        this.scoreFunc = scoreFunc;
    }

    @Override
    public SelectKBest fit(Matrix X, Matrix y) {
        return fit(X, y.col(0));
    }

    public SelectKBest fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;
        int actualK = Math.min(k, m);

        scores = new double[m];
        pvalues = new double[m];

        if ("f_classif".equals(scoreFunc)) {
            computeFClassif(X, y, n);
        } else if ("f_regression".equals(scoreFunc)) {
            computeFRegression(X, y, n);
        } else if ("chi2".equals(scoreFunc)) {
            computeChi2(X, y, n);
        } else {
            throw new IllegalArgumentException("Unsupported score function: " + scoreFunc);
        }

        Integer[] idx = new Integer[m];
        for (int i = 0; i < m; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(scores[b], scores[a]));

        selectedIndices = new int[actualK];
        for (int i = 0; i < actualK; i++) {
            selectedIndices[i] = idx[i];
        }
        Arrays.sort(selectedIndices);

        this.fitted = true;
        return this;
    }

    private void computeFClassif(Matrix X, Vector y, int n) {
        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) y.get(i));
        }
        int[] classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        int nClasses = classes.length;

        double[] globalMean = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += X.get(i, j);
            }
            globalMean[j] = sum / n;
        }

        double[][] classMeans = new double[nClasses][nFeatures];
        int[] classCounts = new int[nClasses];

        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            classCounts[c]++;
            for (int j = 0; j < nFeatures; j++) {
                classMeans[c][j] += X.get(i, j);
            }
        }
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < nFeatures; j++) {
                classMeans[c][j] /= classCounts[c];
            }
        }

        for (int j = 0; j < nFeatures; j++) {
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
    }

    private void computeFRegression(Matrix X, Vector y, int n) {
        double yMean = y.mean();
        for (int j = 0; j < nFeatures; j++) {
            double xMean = 0;
            for (int i = 0; i < n; i++) {
                xMean += X.get(i, j);
            }
            xMean /= n;

            double ssReg = 0, ssRes = 0;
            for (int i = 0; i < n; i++) {
                ssReg += (X.get(i, j) - xMean) * (y.get(i) - yMean);
                ssRes += (y.get(i) - yMean) * (y.get(i) - yMean);
            }
            double numer = ssReg * ssReg;
            double denom = 0;
            for (int i = 0; i < n; i++) {
                denom += (X.get(i, j) - xMean) * (X.get(i, j) - xMean);
            }
            double fStat = denom > 0 && ssRes > 0
                ? (numer / denom) / (ssRes / (n - 2))
                : 0;
            scores[j] = fStat;
            pvalues[j] = fStat > 0 ? fDistributionPValue(fStat, 1, n - 2) : 1.0;
        }
    }

    private void computeChi2(Matrix X, Vector y, int n) {
        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) y.get(i));
        }
        int[] classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        int nClasses = classes.length;
        int[] classCounts = new int[nClasses];
        for (int i = 0; i < n; i++) {
            classCounts[indexOf(classes, (int) y.get(i))]++;
        }

        for (int j = 0; j < nFeatures; j++) {
            double chi2 = 0;
            double featureSum = 0;
            for (int i = 0; i < n; i++) {
                featureSum += X.get(i, j);
            }

            for (int c = 0; c < nClasses; c++) {
                double observed = 0;
                for (int i = 0; i < n; i++) {
                    if ((int) y.get(i) == classes[c]) {
                        observed += X.get(i, j);
                    }
                }
                double expected = featureSum * classCounts[c] / n;
                if (expected > 0) {
                    double diff = observed - expected;
                    chi2 += diff * diff / expected;
                }
            }
            scores[j] = chi2;
            pvalues[j] = chi2DistributionPValue(chi2, nClasses - 1);
        }
    }

    private double fDistributionPValue(double f, int df1, int df2) {
        if (df2 <= 0 || f < 0) {
            return 1.0;
        }
        double x = (double) df1 * f / (df1 * f + df2);
        return 1.0 - regularizedIncompleteBeta(df1 / 2.0, df2 / 2.0, x);
    }

    private double chi2DistributionPValue(double chi2, int df) {
        if (df <= 0 || chi2 < 0) {
            return 1.0;
        }
        return 1.0 - regularizedIncompleteGamma(df / 2.0, chi2 / 2.0);
    }

    private double regularizedIncompleteGamma(double a, double x) {
        if (x < 0 || a <= 0) {
            return 0;
        }
        if (x < a + 1) {
            return seriesGamma(a, x);
        }
        return 1.0 - continuedFractionGamma(a, x);
    }

    private double seriesGamma(double a, double x) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 1; n <= 200; n++) {
            ap += 1;
            del *= x / ap;
            sum += del;
            if (Math.abs(del) < Math.abs(sum) * 1e-15) {
                break;
            }
        }
        return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
    }

    private double continuedFractionGamma(double a, double x) {
        double b = x + 1 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= 200; i++) {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = b + an / c;
            if (Math.abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-15) {
                break;
            }
        }
        return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
    }

    private double regularizedIncompleteBeta(double a, double b, double x) {
        if (x < 0 || x > 1) {
            return 0;
        }
        if (x == 0 || x == 1) {
            return x == 0 ? 0 : 1;
        }
        double bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b)
            + a * Math.log(x) + b * Math.log(1 - x));
        if (x < (a + 1) / (a + b + 2)) {
            return bt * betaCF(a, b, x) / a;
        }
        return 1 - bt * betaCF(b, a, 1 - x) / b;
    }

    private double betaCF(double a, double b, double x) {
        double qab = a + b, qap = a + 1, qam = a - 1;
        double c = 1.0, d = 1.0 - qab * x / qap;
        if (Math.abs(d) < 1e-30) {
            d = 1e-30;
        }
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 200; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-15) {
                break;
            }
        }
        return h;
    }

    private double lgamma(double x) {
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

    @Override
    public Matrix transform(Matrix X) {
        Validation.checkFitted(fitted, "SelectKBest");
        Matrix result = new Matrix(X.rows(), selectedIndices.length);
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                result.set(i, j, X.get(i, selectedIndices[j]));
            }
        }
        return result;
    }

    @Override
    public Matrix inverseTransform(Matrix X) {
        throw new UnsupportedOperationException("SelectKBest does not support inverseTransform");
    }

    @Override
    public Matrix fitTransform(Matrix X, Matrix y) {
        return fit(X, y).transform(X);
    }

    public Matrix fitTransform(Matrix X, Vector y) {
        return fit(X, y).transform(X);
    }

    public int[] getSupport() {
        return selectedIndices;
    }

    public int[] getSelectedIndices() {
        return selectedIndices;
    }

    public double[] getScores() {
        return scores;
    }

    public double[] getPValues() {
        return pvalues;
    }

    public boolean isFitted() {
        return fitted;
    }

    private int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("k", k);
        p.put("score_func", scoreFunc);
        return Collections.unmodifiableMap(p);
    }
}
