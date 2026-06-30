package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * AdaBoost regressor.
 *
 * <p>Fits a sequence of weak regressors on weighted versions of the training
 * data using the AdaBoost.R2 algorithm. Prediction uses weighted median.
 *
 * <p>Mirrors {@code sklearn.ensemble.AdaBoostRegressor}.
 *
 * <p>Usage:
 * <pre>{@code
 * AdaBoostRegressor ada = new AdaBoostRegressor(50, 3, 42);
 * ada.fit(X, y);
 * Vector preds = ada.predict(X_test);
 * }</pre>
 */
public class AdaBoostRegressor implements Predictor<Matrix, Vector, Vector> {

    private int nEstimators;
    private int maxDepth;
    private double learningRate;
    private long seed;
    private boolean fitted;
    private List<DecisionTreeRegressor> estimators;
    private double[] estimatorWeights;
    private int nFeatures;

    /**
     * Create an AdaBoost regressor.
     *
     * @param nEstimators number of weak learners
     * @param maxDepth    max depth of each weak learner
     * @param seed        random seed
     */
    public AdaBoostRegressor(int nEstimators, int maxDepth, long seed) {
        this(nEstimators, maxDepth, 1.0, seed);
    }

    /**
     * Create an AdaBoost regressor with full control.
     *
     * @param nEstimators  number of weak learners
     * @param maxDepth     max depth of each weak learner
     * @param learningRate learning rate
     * @param seed         random seed
     */
    public AdaBoostRegressor(int nEstimators, int maxDepth,
                              double learningRate, long seed) {
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be >= 1");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.learningRate = learningRate;
        this.seed = seed;
    }

    @Override
    public AdaBoostRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        double[] sampleWeights = new double[n];
        Arrays.fill(sampleWeights, 1.0 / n);

        estimators = new ArrayList<>();
        estimatorWeights = new double[nEstimators];

        for (int t = 0; t < nEstimators; t++) {
            Object[] bootData = bootstrapSample(X, y, sampleWeights, t);
            DecisionTreeRegressor stump = new DecisionTreeRegressor(
                maxDepth, 2, 1);
            stump.fit((Matrix) bootData[0], (Vector) bootData[1]);
            estimators.add(stump);

            Vector preds = stump.predict(X);

            double maxError = 0;
            for (int i = 0; i < n; i++) {
                double err = Math.abs(preds.get(i) - y.get(i));
                if (err > maxError) {
                    maxError = err;
                }
            }

            if (maxError == 0) {
                estimatorWeights[t] = 1.0;
                break;
            }

            double[] adjustedErrors = new double[n];
            double weightedError = 0;
            for (int i = 0; i < n; i++) {
                adjustedErrors[i] = Math.abs(preds.get(i) - y.get(i)) / maxError;
                weightedError += sampleWeights[i] * adjustedErrors[i];
            }

            if (weightedError >= 0.5) {
                estimatorWeights[t] = 0;
                continue;
            }

            double beta = weightedError / Math.max(1 - weightedError, 1e-10);
            double alpha = learningRate * Math.log(1.0 / beta);
            estimatorWeights[t] = alpha;

            double sumW = 0;
            for (int i = 0; i < n; i++) {
                sampleWeights[i] *= Math.pow(beta, 1 - adjustedErrors[i]);
                sumW += sampleWeights[i];
            }
            for (int i = 0; i < n; i++) {
                sampleWeights[i] /= sumW;
            }
        }

        fitted = true;
        return this;
    }

    private Object[] bootstrapSample(Matrix X, Vector y, double[] weights, long iterSeed) {
        int n = X.rows();
        int m = X.cols();
        RandomGenerator rng = new RandomGenerator(seed + iterSeed * 1000);

        double[] cumSum = new double[n];
        cumSum[0] = weights[0];
        for (int i = 1; i < n; i++) {
            cumSum[i] = cumSum[i - 1] + weights[i];
        }
        double totalW = cumSum[n - 1];

        Matrix bootX = new Matrix(n, m);
        Vector bootY = new Vector(n);
        for (int i = 0; i < n; i++) {
            double r = rng.nextDouble() * totalW;
            int idx = lowerBound(cumSum, r);
            idx = Math.min(idx, n - 1);
            for (int j = 0; j < m; j++) {
                bootX.set(i, j, X.get(idx, j));
            }
            bootY.set(i, y.get(idx));
        }
        return new Object[]{bootX, bootY};
    }

    private int lowerBound(double[] arr, double val) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (arr[mid] < val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "AdaBoostRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int nEst = estimators.size();
        double[] out = new double[n];

        for (int i = 0; i < n; i++) {
            double[][] rowData = new double[1][nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                rowData[0][j] = X.get(i, j);
            }
            Matrix rowX = new Matrix(rowData);

            double[][] weightedPreds = new double[nEst][2];
            int count = 0;
            double totalWeight = 0;
            for (int t = 0; t < nEst; t++) {
                if (estimatorWeights[t] == 0) {
                    continue;
                }
                double pred = estimators.get(t).predict(rowX).get(0);
                weightedPreds[count][0] = pred;
                weightedPreds[count][1] = estimatorWeights[t];
                totalWeight += estimatorWeights[t];
                count++;
            }

            if (count == 0) {
                out[i] = 0;
                continue;
            }

            // Sort by prediction value
            Arrays.sort(weightedPreds, 0, count, (a, b) -> Double.compare(a[0], b[0]));

            // Weighted median
            double half = totalWeight / 2.0;
            double cumW = 0;
            out[i] = weightedPreds[count - 1][0];
            for (int j = 0; j < count; j++) {
                cumW += weightedPreds[j][1];
                if (cumW >= half) {
                    out[i] = weightedPreds[j][0];
                    break;
                }
            }
        }

        return new Vector(out);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "AdaBoostRegressor");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        double ssRes = 0, ssTot = 0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_estimators", nEstimators);
        params.put("max_depth", maxDepth);
        params.put("learning_rate", learningRate);
        return Collections.unmodifiableMap(params);
    }
}
