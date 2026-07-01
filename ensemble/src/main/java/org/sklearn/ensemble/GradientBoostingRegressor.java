package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Gradient Boosting for regression.
 *
 * <p>Builds an additive ensemble of regression trees optimizing
 * a differentiable loss function (squared error, absolute error, huber).
 *
 * <p>Mirrors {@code sklearn.ensemble.GradientBoostingRegressor}.
 */
public class GradientBoostingRegressor implements Predictor<Matrix, Vector, Vector> {

    private String loss;
    private double learningRate;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private double subsample;
    private double alpha;
    private long seed;

    private List<DecisionTreeRegressor> estimators;
    private double initPrediction;
    private double yMean;

    public GradientBoostingRegressor() {
        this("ls", 0.1, 100, 3, 2, 1, 1.0, 0.9, 42);
    }

    public GradientBoostingRegressor(int nEstimators, int maxDepth, long seed) {
        this("ls", 0.1, nEstimators, maxDepth, 2, 1, 1.0, 0.9, seed);
    }

    public GradientBoostingRegressor(String loss, double learningRate, int nEstimators,
                                      int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                                      double subsample, double alpha, long seed) {
        this.loss = loss;
        this.learningRate = learningRate;
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.subsample = subsample;
        this.alpha = alpha;
        this.seed = seed;
    }

    @Override
    public GradientBoostingRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();

        yMean = y.mean();
        initPrediction = yMean;

        double[] rawPred = new double[n];
        Arrays.fill(rawPred, initPrediction);

        estimators = new ArrayList<>();
        Random rng = new Random(seed);

        for (int iter = 0; iter < nEstimators; iter++) {
            boolean[] sampleMask = new boolean[n];
            if (subsample < 1.0) {
                int subSize = Math.max(1, (int) Math.round(n * subsample));
                List<Integer> pool = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    pool.add(i);
                }
                Collections.shuffle(pool, rng);
                for (int i = 0; i < subSize; i++) {
                    sampleMask[pool.get(i)] = true;
                }
            } else {
                Arrays.fill(sampleMask, true);
            }

            // Compute negative gradient based on loss type
            double[] negGrad = new double[n];
            switch (loss) {
                case "ls":
                    for (int i = 0; i < n; i++) {
                        negGrad[i] = y.get(i) - rawPred[i];
                    }
                    break;
                case "lad":
                    for (int i = 0; i < n; i++) {
                        negGrad[i] = Math.signum(y.get(i) - rawPred[i]);
                    }
                    break;
                case "huber":
                    double delta = huberDelta(y, rawPred, alpha);
                    for (int i = 0; i < n; i++) {
                        double diff = y.get(i) - rawPred[i];
                        if (Math.abs(diff) <= delta) {
                            negGrad[i] = diff;
                        } else {
                            negGrad[i] = delta * Math.signum(diff);
                        }
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported loss: " + loss);
            }

            // Count subset and build fit data
            int subN = 0;
            for (boolean b : sampleMask) {
                if (b) {
                    subN++;
                }
            }
            double[][] fitData = new double[subN][m];
            double[] fitTarget = new double[subN];
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (sampleMask[i]) {
                    for (int j = 0; j < m; j++) {
                        fitData[idx][j] = X.get(i, j);
                    }
                    fitTarget[idx] = negGrad[i];
                    idx++;
                }
            }

            DecisionTreeRegressor tree = new DecisionTreeRegressor(
                maxDepth, minSamplesSplit, minSamplesLeaf, false, seed + iter);
            tree.fit(new Matrix(fitData), new Vector(fitTarget));

            // Predict on full data
            double[][] allData = new double[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    allData[i][j] = X.get(i, j);
                }
            }
            Vector treePred = tree.predict(new Matrix(allData));
            for (int i = 0; i < n; i++) {
                rawPred[i] += learningRate * treePred.get(i);
            }

            estimators.add(tree);
        }

        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        int n = X.rows();
        int m = X.cols();
        double[][] xData = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                xData[i][j] = X.get(i, j);
            }
        }
        Matrix xMat = new Matrix(xData);

        double[] preds = new double[n];
        Arrays.fill(preds, initPrediction);

        for (DecisionTreeRegressor tree : estimators) {
            Vector treePred = tree.predict(xMat);
            for (int i = 0; i < n; i++) {
                preds[i] += learningRate * treePred.get(i);
            }
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        double ssRes = 0;
        double ssTot = 0;
        double yMeanVal = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMeanVal;
            ssTot += diffMean * diffMean;
        }
        return ssTot > 0 ? 1.0 - ssRes / ssTot : 1.0;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss);
        p.put("learning_rate", learningRate);
        p.put("n_estimators", nEstimators);
        p.put("max_depth", maxDepth);
        p.put("min_samples_split", minSamplesSplit);
        p.put("min_samples_leaf", minSamplesLeaf);
        p.put("subsample", subsample);
        p.put("alpha", alpha);
        return p;
    }

    private double huberDelta(Vector y, double[] rawPred, double quantile) {
        int n = y.size();
        double[] absErr = new double[n];
        for (int i = 0; i < n; i++) {
            absErr[i] = Math.abs(y.get(i) - rawPred[i]);
        }
        Arrays.sort(absErr);
        int idx = (int) Math.round(quantile * n);
        idx = Math.min(idx, n - 1);
        return absErr[idx];
    }
}
