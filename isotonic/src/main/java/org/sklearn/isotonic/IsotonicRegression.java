package org.sklearn.isotonic;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Isotonic regression.
 *
 * <p>Fits a non-decreasing piecewise constant function to the data
 * using the pool adjacent violators algorithm (PAVA).
 *
 * <p>Mirrors {@code sklearn.isotonic.IsotonicRegression}.
 */
public class IsotonicRegression implements Predictor<Matrix, Vector, Vector> {

    private boolean increasing;
    private boolean fitted;
    private double[] xThresholds;
    private double[] yValues;
    private int nFeatures;

    /**
     * Create IsotonicRegression with default params.
     */
    public IsotonicRegression() {
        this(true);
    }

    /**
     * Create IsotonicRegression.
     *
     * @param increasing whether to fit increasing (true) or decreasing (false)
     */
    public IsotonicRegression(boolean increasing) {
        this.increasing = increasing;
    }

    @Override
    public IsotonicRegression fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        if (X.cols() != 1) {
            throw new IllegalArgumentException("IsotonicRegression only supports 1 feature, got " + X.cols());
        }
        nFeatures = 1;
        int n = X.rows();

        double[][] pairs = new double[n][2];
        for (int i = 0; i < n; i++) {
            pairs[i][0] = X.get(i, 0);
            pairs[i][1] = y.get(i);
        }
        Arrays.sort(pairs, (a, b) -> Double.compare(a[0], b[0]));

        double[] x = new double[n];
        double[] yVal = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = pairs[i][0];
            yVal[i] = pairs[i][1];
        }

        // PAVA
        List<Block> blocks = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            blocks.add(new Block(yVal[i], x[i], x[i], 1));
        }

        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < blocks.size() - 1; i++) {
                Block a = blocks.get(i);
                Block b = blocks.get(i + 1);
                if ((increasing && a.getValue() > b.getValue())
                        || (!increasing && a.getValue() < b.getValue())) {
                    double mergedVal = (a.getValue() * a.getCount() + b.getValue() * b.getCount())
                            / (a.getCount() + b.getCount());
                    blocks.set(i, new Block(mergedVal, a.getXMin(), b.getXMax(),
                            a.getCount() + b.getCount()));
                    blocks.remove(i + 1);
                    changed = true;
                    break;
                }
            }
        }

        xThresholds = new double[blocks.size() + 1];
        yValues = new double[blocks.size()];
        for (int i = 0; i < blocks.size(); i++) {
            Block b = blocks.get(i);
            xThresholds[i] = b.getXMin();
            yValues[i] = b.getValue();
        }
        xThresholds[blocks.size()] = blocks.get(blocks.size() - 1).getXMax();

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "IsotonicRegression");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " feature, got " + X.cols());
        }
        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            double x = X.get(i, 0);
            int idx = Arrays.binarySearch(xThresholds, x);
            if (idx < 0) {
                idx = -idx - 2;
            }
            idx = Math.max(0, Math.min(idx, yValues.length - 1));
            preds[i] = yValues[idx];
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return org.sklearn.metrics.RegressionMetrics.r2Score(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("increasing", increasing);
        params.put("x_thresholds_", xThresholds);
        params.put("y_thresholds_", yValues);
        return params;
    }

    public double[] getXThresholds() { return xThresholds; }
    public double[] getYValues() { return yValues; }
    public boolean isFitted() { return fitted; }

    private static class Block {
        private final double value;
        private final double xMin;
        private final double xMax;
        private final int count;

        Block(double value, double xMin, double xMax, int count) {
            this.value = value;
            this.xMin = xMin;
            this.xMax = xMax;
            this.count = count;
        }

        double getValue() { return value; }
        double getXMin() { return xMin; }
        double getXMax() { return xMax; }
        int getCount() { return count; }
    }
}
