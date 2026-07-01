package org.sklearn.model_selection;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Grid search cross-validation.
 *
 * <p>Exhaustive search over specified parameter values for an estimator.
 *
 * <p>Mirrors {@code sklearn.model_selection.GridSearchCV}.
 */
public class GridSearchCV implements Predictor<Matrix, Vector, Vector> {

    private Estimator<Matrix, Vector> baseEstimator;
    private Map<String, double[]> paramGrid;
    private KFold cv;
    private boolean stratified;
    private boolean refit;
    private Map<String, Double> bestParams;
    private double bestScore;
    private Estimator<Matrix, Vector> bestEstimator;
    private List<Map<String, Double>> cvResults;

    /**
     * Create GridSearchCV.
     *
     * @param baseEstimator the estimator to tune
     * @param paramGrid     map of parameter names to arrays of values to try
     * @param cv            number of folds (KFold is created internally)
     * @param stratified    whether to use StratifiedKFold
     * @param refit         whether to refit best estimator on full data
     */
    public GridSearchCV(Estimator<Matrix, Vector> baseEstimator,
                        Map<String, double[]> paramGrid,
                        int cv, boolean stratified, boolean refit) {
        this.baseEstimator = baseEstimator;
        this.paramGrid = paramGrid;
        this.cv = new KFold(cv, false, 42);
        this.stratified = stratified;
        this.refit = refit;
    }

    @Override
    public GridSearchCV fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        List<String> paramNames = new ArrayList<>(paramGrid.keySet());
        List<double[]> paramValues = new ArrayList<>();
        for (String name : paramNames) {
            paramValues.add(paramGrid.get(name));
        }

        List<List<Double>> combinations = new ArrayList<>();
        cartesianProduct(paramValues, 0, new ArrayList<>(), combinations);

        double foundBestScore = -Double.MAX_VALUE;
        Map<String, Double> foundBestParams = null;
        Estimator<Matrix, Vector> foundBestEstimator = null;
        cvResults = new ArrayList<>();

        for (List<Double> combo : combinations) {
            Estimator<Matrix, Vector> est = cloneEstimator(baseEstimator);
            Map<String, Double> params = new LinkedHashMap<>();

            for (int i = 0; i < paramNames.size(); i++) {
                params.put(paramNames.get(i), combo.get(i));
                setField(est, paramNames.get(i), combo.get(i));
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

            params.put("score", score);
            cvResults.add(new LinkedHashMap<>(params));

            if (score > foundBestScore) {
                foundBestScore = score;
                foundBestParams = new LinkedHashMap<>();
                for (int i = 0; i < paramNames.size(); i++) {
                    foundBestParams.put(paramNames.get(i), combo.get(i));
                }
                foundBestEstimator = est;
            }
        }

        this.bestParams = foundBestParams;
        this.bestScore = foundBestScore;
        this.bestEstimator = foundBestEstimator;

        if (refit && bestEstimator != null) {
            bestEstimator.fit(X, y);
        }

        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        if (bestEstimator == null) {
            throw new IllegalStateException("GridSearchCV has not been fitted yet");
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
        params.put("cv", cv.getNSplits());
        params.put("stratified", stratified);
        params.put("refit", refit);
        return params;
    }

    /**
     * Returns the best parameters found.
     */
    public Map<String, Double> getBestParams() {
        return bestParams;
    }

    /**
     * Returns the best cross-validation score.
     */
    public double getBestScore() {
        return bestScore;
    }

    /**
     * Returns the best estimator.
     */
    public Estimator<Matrix, Vector> getBestEstimator() {
        return bestEstimator;
    }

    /**
     * Returns detailed CV results.
     */
    public List<Map<String, Double>> getCvResults() {
        return cvResults;
    }

    private void cartesianProduct(List<double[]> values, int depth,
                                   List<Double> current, List<List<Double>> result) {
        if (depth == values.size()) {
            result.add(new ArrayList<>(current));
            return;
        }
        for (double v : values.get(depth)) {
            current.add(v);
            cartesianProduct(values, depth + 1, current, result);
            current.remove(current.size() - 1);
        }
    }

    private void setField(Object obj, String name, double value) {
        try {
            Field field = obj.getClass().getField(name);
            Class<?> type = field.getType();
            if (type == int.class || type == Integer.class) {
                field.setInt(obj, (int) Math.round(value));
            } else if (type == long.class || type == Long.class) {
                field.setLong(obj, Math.round(value));
            } else if (type == float.class || type == Float.class) {
                field.setFloat(obj, (float) value);
            } else if (type == boolean.class || type == Boolean.class) {
                field.setBoolean(obj, value != 0);
            } else {
                field.setDouble(obj, value);
            }
        } catch (Exception e) {
            throw new RuntimeException("Cannot set field " + name + " on " + obj.getClass().getName(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private Estimator<Matrix, Vector> cloneEstimator(Estimator<Matrix, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
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
