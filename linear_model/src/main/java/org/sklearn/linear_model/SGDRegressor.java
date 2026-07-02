package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * SGD regressor.
 */
public class SGDRegressor implements Predictor<Matrix, Vector, Vector> {

    private String loss;
    private String penalty;
    private double alpha;
    private double l1Ratio;
    private boolean fitIntercept;
    private int maxIter;
    private double tol;
    private String learningRate;
    private double eta0;
    private double powerT;
    private double epsilon;
    private boolean shuffle;
    private long randomState;
    private boolean average;
    private boolean warmStart;

    private Vector coef;
    private double intercept;
    private boolean fitted;
    private int nFeatures;
    private int nIter;

    public SGDRegressor() {
        this("squared_error", "l2", 0.0001, 0.15, true, 1000, 1e-3,
            "invscaling", 0.01, 0.25, 0.1, true, 42, false, false);
    }

    @SuppressWarnings("checkstyle:parameternumber")
    public SGDRegressor(String loss, String penalty, double alpha, double l1Ratio,
                         boolean fitIntercept, int maxIter, double tol,
                         String learningRate, double eta0, double powerT,
                         double epsilon, boolean shuffle, long randomState,
                         boolean average, boolean warmStart) {
        this.loss = loss;
        this.penalty = penalty;
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.learningRate = learningRate;
        this.eta0 = eta0;
        this.powerT = powerT;
        this.epsilon = epsilon;
        this.shuffle = shuffle;
        this.randomState = randomState;
        this.average = average;
        this.warmStart = warmStart;
    }

    @Override
    public SGDRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        double[] w = new double[m];
        double bias = 0.0;
        if (warmStart && coef != null) {
            for (int j = 0; j < m; j++) {
                w[j] = coef.get(j);
            }
            bias = intercept;
        }

        int t = 1;
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
                for (int j = 0; j < m; j++) {
                    pred += X.get(i, j) * w[j];
                }

                double residual = pred - y.get(i);
                double dloss = lossGradReg(residual, pred, y.get(i));
                double eta = learningRate(t);

                boolean isL1 = penalty.equals("l1") || penalty.equals("elasticnet");
                double l1Mult = penalty.equals("l1") ? 1.0 : l1Ratio;

                for (int j = 0; j < m; j++) {
                    if (penalty.equals("l2")) {
                        w[j] *= (1.0 - eta * alpha);
                    } else if (penalty.equals("elasticnet")) {
                        w[j] *= (1.0 - eta * alpha * (1.0 - l1Ratio));
                    }
                    w[j] -= eta * dloss * X.get(i, j);

                    if (isL1 && eta * alpha * l1Mult > Math.abs(w[j])) {
                        w[j] = 0.0;
                    } else if (isL1 && w[j] > 0) {
                        w[j] -= eta * alpha * l1Mult;
                    } else if (isL1) {
                        w[j] += eta * alpha * l1Mult;
                    }
                }
                bias -= eta * dloss;
                t++;
            }
        }

        this.coef = new Vector(w);
        this.intercept = bias;
        this.nIter = maxIter;
        this.fitted = true;
        return this;
    }

    private double lossGradReg(double residual, double pred, double y) {
        switch (loss) {
            case "squared_error":
                return residual;
            case "huber":
                double absRes = Math.abs(residual);
                if (absRes <= epsilon) {
                    return residual;
                }
                return epsilon * Math.signum(residual);
            case "epsilon_insensitive":
                if (residual > epsilon) {
                    return 1.0;
                }
                if (residual < -epsilon) {
                    return -1.0;
                }
                return 0.0;
            case "squared_epsilon_insensitive":
                if (residual > epsilon) {
                    return 2.0 * (residual - epsilon);
                }
                if (residual < -epsilon) {
                    return 2.0 * (residual + epsilon);
                }
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown loss: " + loss);
        }
    }

    private double learningRate(int t) {
        switch (learningRate) {
            case "constant": return eta0 > 0 ? eta0 : 0.01;
            case "invscaling": return eta0 / Math.pow(t, powerT);
            case "optimal": return 1.0 / (alpha * (1.0 + t - 1.0));
            default: return 0.01;
        }
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "SGDRegressor");
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
        Validation.checkFitted(fitted, "SGDRegressor");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

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
    public int getNIter() {
        return nIter;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss); p.put("penalty", penalty); p.put("alpha", alpha);
        p.put("l1_ratio", l1Ratio); p.put("fit_intercept", fitIntercept);
        p.put("max_iter", maxIter); p.put("tol", tol);
        p.put("learning_rate", learningRate); p.put("eta0", eta0);
        p.put("power_t", powerT); p.put("epsilon", epsilon);
        return Collections.unmodifiableMap(p);
    }
}
