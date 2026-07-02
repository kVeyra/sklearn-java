package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * RANSAC (RANdom SAmple Consensus) regressor.
 *
 * <p>Fits a robust regression model by iteratively selecting random
 * subsets, fitting a base model, and identifying inliers.
 *
 * <p>Mirrors {@code sklearn.linear_model.RANSACRegressor}.
 */
public class RANSACRegressor implements Predictor<Matrix, Vector, Vector> {

    private Predictor<Matrix, Vector, Vector> baseEstimator;
    private int minSamples;
    private int maxTrials;
    private double residualThreshold;
    private long randomState;
    private boolean fitted;
    private Predictor<Matrix, Vector, Vector> bestEstimator;
    private int[] inlierMask;
    private int nFeatures;

    /**
     * Create RANSACRegressor with default params.
     */
    public RANSACRegressor() {
        this(null, 0, 100, Double.NaN, 42);
    }

    /**
     * Create RANSACRegressor.
     */
    public RANSACRegressor(Predictor<Matrix, Vector, Vector> baseEstimator,
                           int minSamples, int maxTrials,
                           double residualThreshold, long randomState) {
        this.baseEstimator = baseEstimator != null ? baseEstimator : new LinearRegression();
        this.minSamples = minSamples;
        this.maxTrials = maxTrials;
        this.residualThreshold = residualThreshold;
        this.randomState = randomState;
    }

    @Override
    public RANSACRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();
        int n = X.rows();
        int minS = minSamples > 0 ? minSamples : Math.max(1, nFeatures + 1);

        double threshold = Double.isNaN(residualThreshold) ? computeMedianAbsDev(y) : residualThreshold;

        Random rng = new Random(randomState);
        double bestScore = -Double.MAX_VALUE;
        int[] bestInliers = null;
        Predictor<Matrix, Vector, Vector> bestEst = null;

        for (int trial = 0; trial < maxTrials; trial++) {
            int[] subset = selectRandomSubset(n, minS, rng);
            Matrix xSub = extractRows(X, subset);
            Vector ySub = extractRows(y, subset);

            Predictor<Matrix, Vector, Vector> est = cloneEstimator(baseEstimator);
            est.fit(xSub, ySub);

            Vector preds = est.predict(X);
            boolean[] inlierBool = new boolean[n];
            int inlierCount = 0;
            for (int i = 0; i < n; i++) {
                double residual = Math.abs(preds.get(i) - y.get(i));
                if (residual < threshold) {
                    inlierBool[i] = true;
                    inlierCount++;
                }
            }

            if (inlierCount < minS) continue;

            int[] inlierIdx = new int[inlierCount];
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (inlierBool[i]) inlierIdx[idx++] = i;
            }

            Matrix xInlier = extractRows(X, inlierIdx);
            Vector yInlier = extractRows(y, inlierIdx);

            Predictor<Matrix, Vector, Vector> refinedEst = cloneEstimator(baseEstimator);
            refinedEst.fit(xInlier, yInlier);
            double score = refinedEst.score(X, y);

            if (score > bestScore) {
                bestScore = score;
                bestEst = refinedEst;
                bestInliers = inlierIdx;
            }
        }

        if (bestEst == null) {
            bestEst = baseEstimator;
            bestEst.fit(X, y);
            bestInliers = new int[n];
            for (int i = 0; i < n; i++) bestInliers[i] = i;
        }

        bestEstimator = bestEst;
        inlierMask = bestInliers;
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "RANSACRegressor");
        return bestEstimator.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        return bestEstimator.score(X, y);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("max_trials", maxTrials);
        params.put("inlier_mask_", inlierMask);
        return params;
    }

    public int[] getInlierMask() { return inlierMask; }
    public Predictor<Matrix, Vector, Vector> getEstimator() { return bestEstimator; }

    private int[] selectRandomSubset(int n, int k, Random rng) {
        Set<Integer> set = new HashSet<>();
        while (set.size() < k) {
            set.add(rng.nextInt(n));
        }
        return set.stream().mapToInt(Integer::intValue).toArray();
    }

    private static double computeMedianAbsDev(Vector y) {
        int n = y.size();
        double median = 0;
        double[] sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = y.get(i);
        Arrays.sort(sorted);
        median = sorted[n / 2];
        double[] absDev = new double[n];
        for (int i = 0; i < n; i++) absDev[i] = Math.abs(y.get(i) - median);
        Arrays.sort(absDev);
        return absDev[n / 2];
    }

    @SuppressWarnings("unchecked")
    private Predictor<Matrix, Vector, Vector> cloneEstimator(
            Predictor<Matrix, Vector, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator", e);
        }
    }

    private static Matrix extractRows(Matrix X, int[] indices) {
        int n = indices.length;
        int m = X.cols();
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = X.get(indices[i], j);
            }
        }
        return new Matrix(data);
    }

    private static Vector extractRows(Vector y, int[] indices) {
        int n = indices.length;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = y.get(indices[i]);
        }
        return new Vector(data);
    }
}
