package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Bagging regressor.
 *
 * <p>Fits base regressors on random subsets of the original dataset
 * and aggregates predictions by averaging.
 *
 * <p>Mirrors {@code sklearn.ensemble.BaggingRegressor}.
 */
public class BaggingRegressor implements Predictor<Matrix, Vector, Vector> {

    private Estimator<Matrix, Vector> baseEstimator;
    private int nEstimators;
    private double maxSamples;
    private double maxFeatures;
    private boolean bootstrap;
    private boolean bootstrapFeatures;
    private long seed;

    private List<Predictor<Matrix, Vector, Vector>> estimators;

    public BaggingRegressor(Estimator<Matrix, Vector> baseEstimator, int nEstimators, long seed) {
        this(baseEstimator, nEstimators, 1.0, 1.0, true, false, seed);
    }

    public BaggingRegressor(Estimator<Matrix, Vector> baseEstimator, int nEstimators,
                             double maxSamples, double maxFeatures,
                             boolean bootstrap, boolean bootstrapFeatures, long seed) {
        this.baseEstimator = baseEstimator;
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.bootstrapFeatures = bootstrapFeatures;
        this.seed = seed;
    }

    @Override
    @SuppressWarnings("unchecked")
    public BaggingRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        estimators = new ArrayList<>();
        Random rng = new Random(seed);

        for (int t = 0; t < nEstimators; t++) {
            Random rngT = new Random(seed + t * 1000L);
            int sampleSize = (int) Math.round(n * maxSamples);
            int[] sampleIdx = new int[n];
            if (bootstrap) {
                for (int i = 0; i < n; i++) {
                    sampleIdx[i] = rngT.nextInt(n);
                }
            } else {
                List<Integer> pool = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    pool.add(i);
                }
                Collections.shuffle(pool, rngT);
                for (int i = 0; i < Math.min(sampleSize, n); i++) {
                    sampleIdx[i] = pool.get(i);
                }
                if (sampleSize > n) {
                    for (int i = n; i < sampleSize; i++) {
                        sampleIdx[i] = rngT.nextInt(n);
                    }
                }
            }

            int featSize = (int) Math.round(m * maxFeatures);
            featSize = Math.max(1, Math.min(featSize, m));
            int[] featIdx = new int[m];
            for (int i = 0; i < m; i++) {
                featIdx[i] = i;
            }
            if (bootstrapFeatures) {
                for (int i = 0; i < m; i++) {
                    featIdx[i] = rngT.nextInt(m);
                }
                featSize = m;
            } else if (maxFeatures < 1.0) {
                List<Integer> pool = new ArrayList<>();
                for (int i = 0; i < m; i++) {
                    pool.add(i);
                }
                Collections.shuffle(pool, rngT);
                featSize = Math.max(1, featSize);
                for (int i = 0; i < m; i++) {
                    if (i < featSize) {
                        featIdx[i] = pool.get(i);
                    } else {
                        featIdx[i] = -1;
                    }
                }
            }

            double[][] subData = new double[n][featSize];
            double[] subTarget = new double[n];
            for (int i = 0; i < n; i++) {
                int srcRow = sampleIdx[i];
                subTarget[i] = y.get(srcRow);
                for (int j = 0; j < featSize; j++) {
                    subData[i][j] = X.get(srcRow, featIdx[j]);
                }
            }

            Estimator<Matrix, Vector> clone = cloneEstimator(baseEstimator);
            clone.fit(new Matrix(subData), new Vector(subTarget));

            if (clone instanceof Predictor) {
                estimators.add((Predictor<Matrix, Vector, Vector>) clone);
            }
        }
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        int n = X.rows();
        double[] sum = new double[n];

        for (Predictor<Matrix, Vector, Vector> est : estimators) {
            Vector pred = est.predict(X);
            for (int i = 0; i < n; i++) {
                sum[i] += pred.get(i);
            }
        }

        Vector result = new Vector(n);
        int nEst = estimators.size();
        for (int i = 0; i < n; i++) {
            result.set(i, sum[i] / nEst);
        }
        return result;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        double ssRes = 0;
        double ssTot = 0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        return ssTot > 0 ? 1.0 - ssRes / ssTot : 1.0;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators);
        p.put("max_samples", maxSamples);
        p.put("max_features", maxFeatures);
        p.put("bootstrap", bootstrap);
        p.put("bootstrap_features", bootstrapFeatures);
        return p;
    }

    @SuppressWarnings("unchecked")
    private Estimator<Matrix, Vector> cloneEstimator(Estimator<Matrix, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
        }
    }
}
