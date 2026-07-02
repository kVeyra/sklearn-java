package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Passive Aggressive Regressor.
 */
public class PassiveAggressiveRegressor implements Predictor<Matrix, Vector, Vector> {

    private double c;
    private boolean fitIntercept;
    private int maxIter;
    private double tol;
    private String loss;
    private double epsilon;
    private boolean shuffle;
    private long randomState;

    private Vector coef;
    private double intercept;
    private boolean fitted;
    private int nFeatures;

    public PassiveAggressiveRegressor() {
        this(1.0, true, 1000, 1e-3, "epsilon_insensitive", 0.1, true, 42);
    }

    public PassiveAggressiveRegressor(double C, boolean fitIntercept, int maxIter,
                                       double tol, String loss, double epsilon,
                                       boolean shuffle, long randomState) {
        this.c = C;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.loss = loss;
        this.epsilon = epsilon;
        this.shuffle = shuffle;
        this.randomState = randomState;
    }

    @Override
    public PassiveAggressiveRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        double[] w = new double[m];
        double bias = 0.0;
        String lrType = loss.equals("epsilon_insensitive") ? "pa1" : "pa2";

        for (int epoch = 0; epoch < maxIter; epoch++) {
            List<Integer> idxList = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                idxList.add(i);
            }
            if (shuffle) {
                Collections.shuffle(idxList, new Random(randomState + epoch));
            }

            for (int si = 0; si < n; si++) {
                int i = idxList.get(si);
                double pred = bias;
                double xiNormSq = 0.0;
                for (int j = 0; j < m; j++) {
                    double xij = X.get(i, j);
                    pred += xij * w[j];
                    xiNormSq += xij * xij;
                }
                if (xiNormSq == 0) {
                    continue;
                }

                double residual = pred - y.get(i);
                double absRes = Math.abs(residual);
                double lossVal = loss.equals("epsilon_insensitive")
                    ? Math.max(0, absRes - epsilon)
                    : 0.5 * Math.max(0, absRes - epsilon) * Math.max(0, absRes - epsilon);

                if (lossVal == 0) {
                    continue;
                }

                double tau;
                if (lrType.equals("pa1")) {
                    tau = Math.min(c, lossVal / (xiNormSq + 1e-10));
                } else {
                    tau = lossVal / (xiNormSq + 0.5 / (c + 1e-10));
                }

                double sign = residual > 0 ? 1.0 : -1.0;
                for (int j = 0; j < m; j++) {
                    w[j] += tau * sign * X.get(i, j);
                }
                bias += tau * sign;
            }
        }

        this.coef = new Vector(w);
        this.intercept = bias;
        this.fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "PassiveAggressiveRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = intercept;
            for (int j = 0; j < nFeatures; j++) {
                sum += X.get(i, j) * coef.get(j);
            }
            preds[i] = sum;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "PassiveAggressiveRegressor");
        Vector pred = predict(X);
        double ssRes = 0.0, ssTot = 0.0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public Vector getCoef() {
        return coef;
    }
    public double getIntercept() {
        return intercept;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("C", c); p.put("fit_intercept", fitIntercept);
        p.put("max_iter", maxIter); p.put("tol", tol);
        p.put("loss", loss); p.put("epsilon", epsilon);
        return Collections.unmodifiableMap(p);
    }
}
