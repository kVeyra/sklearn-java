package org.sklearn.model_selection;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Randomized search on hyperparameters.
 *
 * <p>Unlike GridSearchCV which exhaustively searches all combinations,
 * RandomizedSearchCV samples a fixed number of parameter settings from
 * specified distributions.
 *
 * <p>Mirrors {@code sklearn.model_selection.RandomizedSearchCV}.
 */
public class RandomizedSearchCV implements Predictor<Matrix, Vector, Vector> {

    private Estimator<Matrix, Vector> baseEstimator;
    private Map<String, Object[]> paramDistributions;
    private int nIter;
    private KFold cv;
    private boolean stratified;
    private boolean refit;
    private long randomState;
    private Map<String, Object> bestParams;
    private double bestScore;
    private Estimator<Matrix, Vector> bestEstimator;
    private List<Map<String, Object>> cvResults;

    /**
     * Create RandomizedSearchCV.
     *
     * @param baseEstimator      the estimator to tune
     * @param paramDistributions map of parameter names to arrays of candidate values
     * @param nIter              number of parameter settings sampled
     * @param cv                 number of folds
     * @param stratified         whether to use StratifiedKFold
     * @param refit              whether to refit on full data
     * @param randomState        random seed
     */
    public RandomizedSearchCV(Estimator<Matrix, Vector> baseEstimator,
                              Map<String, Object[]> paramDistributions,
                              int nIter, int cv, boolean stratified,
                              boolean refit, long randomState) {
        this.baseEstimator = baseEstimator;
        this.paramDistributions = paramDistributions;
        this.nIter = nIter;
        this.cv = new KFold(cv, false, 42);
        this.stratified = stratified;
        this.refit = refit;
        this.randomState = randomState;
    }

    @Override
    public RandomizedSearchCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        List<String> paramNames = new ArrayList<>(paramDistributions.keySet());
        List<Object[]> paramValues = new ArrayList<>();
        for (String name : paramNames) {
            paramValues.add(paramDistributions.get(name));
        }

        Random rng = new Random(randomState);
        Set<String> sampled = new HashSet<>();
        cvResults = new ArrayList<>();
        double foundBestScore = -Double.MAX_VALUE;

        for (int iter = 0; iter < nIter; iter++) {
            Map<String, Object> params = new LinkedHashMap<>();
            StringBuilder key = new StringBuilder();

            for (int i = 0; i < paramNames.size(); i++) {
                Object[] values = paramValues.get(i);
                int idx = rng.nextInt(values.length);
                Object val = values[idx];
                params.put(paramNames.get(i), val);
                key.append(paramNames.get(i)).append("=").append(val).append("|");
            }

            if (sampled.contains(key.toString())) {
                continue;
            }
            sampled.add(key.toString());

            Estimator<Matrix, Vector> est = cloneEstimator(baseEstimator);
            for (Map.Entry<String, Object> e : params.entrySet()) {
                setField(est, e.getKey(), e.getValue());
            }

            double score;
            if (stratified) {
                StratifiedKFold skf = new StratifiedKFold(cv.getNSplits(), false, 42);
                double[] scores = CrossValidation.crossValScore(est, X, y, skf);
                score = average(scores);
            } else {
                double[] scores = CrossValidation.crossValScore(est, X, y, cv);
                score = average(scores);
            }

            Map<String, Object> resultEntry = new LinkedHashMap<>(params);
            resultEntry.put("score", score);
            cvResults.add(resultEntry);

            if (score > foundBestScore) {
                foundBestScore = score;
                bestParams = new LinkedHashMap<>(params);
                bestEstimator = est;
            }
        }

        this.bestScore = foundBestScore;

        if (refit && bestEstimator != null) {
            bestEstimator.fit(X, y);
        }

        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        if (bestEstimator == null) {
            throw new IllegalStateException("RandomizedSearchCV has not been fitted yet");
        }
        if (bestEstimator instanceof Predictor) {
            @SuppressWarnings("unchecked")
            Predictor<Matrix, Vector, Vector> pred = (Predictor<Matrix, Vector, Vector>) bestEstimator;
            return pred.predict(X);
        }
        throw new IllegalStateException("Best estimator does not implement Predictor");
    }

    @Override
    public double score(Matrix X, Vector y) {
        if (bestEstimator instanceof Predictor) {
            @SuppressWarnings("unchecked")
            Predictor<Matrix, Vector, Vector> pred = (Predictor<Matrix, Vector, Vector>) bestEstimator;
            return pred.score(X, y);
        }
        throw new IllegalStateException("Best estimator does not implement Predictor");
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_iter", nIter);
        params.put("cv", cv.getNSplits());
        params.put("stratified", stratified);
        params.put("refit", refit);
        return params;
    }

    public Map<String, Object> getBestParams() {
        return bestParams;
    }

    public double getBestScore() {
        return bestScore;
    }

    public Estimator<Matrix, Vector> getBestEstimator() {
        return bestEstimator;
    }

    public List<Map<String, Object>> getCvResults() {
        return cvResults;
    }

    @SuppressWarnings("unchecked")
    private Estimator<Matrix, Vector> cloneEstimator(Estimator<Matrix, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
        }
    }

    private void setField(Object obj, String name, Object value) {
        try {
            Field field = obj.getClass().getDeclaredField(name);
            field.setAccessible(true);
            Class<?> type = field.getType();
            if (value instanceof Number) {
                double dv = ((Number) value).doubleValue();
                if (type == int.class || type == Integer.class) {
                    field.setInt(obj, (int) Math.round(dv));
                } else if (type == long.class || type == Long.class) {
                    field.setLong(obj, Math.round(dv));
                } else if (type == float.class || type == Float.class) {
                    field.setFloat(obj, (float) dv);
                } else if (type == double.class || type == Double.class) {
                    field.setDouble(obj, dv);
                } else if (type == boolean.class || type == Boolean.class) {
                    field.setBoolean(obj, dv != 0);
                }
            } else if (value instanceof Boolean) {
                field.setBoolean(obj, (Boolean) value);
            } else {
                field.set(obj, value);
            }
        } catch (Exception e) {
            throw new RuntimeException("Cannot set field " + name + " on " + obj.getClass().getName(), e);
        }
    }

    private double average(double[] scores) {
        double sum = 0;
        for (double s : scores) {
            sum += s;
        }
        return sum / scores.length;
    }
}
