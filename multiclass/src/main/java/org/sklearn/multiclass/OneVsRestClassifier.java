package org.sklearn.multiclass;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * One-vs-the-rest (OvR) multiclass strategy.
 *
 * <p>Fits one binary classifier per class, where each classifier
 * distinguishes one class from all others.
 *
 * <p>Mirrors {@code sklearn.multiclass.OneVsRestClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * OneVsRestClassifier ovr = new OneVsRestClassifier(baseEstimator);
 * ovr.fit(X, y);
 * Vector preds = ovr.predict(X_test);
 * }</pre>
 */
public class OneVsRestClassifier implements Predictor<Matrix, Vector, Vector> {

    private Estimator<Matrix, Vector> baseEstimator;
    private boolean fitted;
    private int[] classes;
    private int nClasses;
    private int nFeatures;
    private List<Estimator<Matrix, Vector>> estimators;

    /**
     * Create OneVsRestClassifier.
     *
     * @param baseEstimator the binary estimator to use for each class
     */
    public OneVsRestClassifier(Estimator<Matrix, Vector> baseEstimator) {
        this.baseEstimator = baseEstimator;
    }

    @Override
    public OneVsRestClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        nFeatures = X.cols();

        Set<Integer> seen = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            seen.add((int) y.get(i));
        }
        classes = seen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(classes);
        nClasses = classes.length;

        estimators = new ArrayList<>();
        for (int c = 0; c < nClasses; c++) {
            int targetClass = classes[c];
            Vector yBinary = new Vector(y.size());
            for (int i = 0; i < y.size(); i++) {
                yBinary.set(i, (int) y.get(i) == targetClass ? 1.0 : 0.0);
            }

            Estimator<Matrix, Vector> estimator = cloneEstimator(baseEstimator);
            estimator.fit(X, yBinary);
            estimators.add(estimator);
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "OneVsRestClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Expected " + nFeatures + " features, got " + X.cols());
        }

        Matrix scores = decisionFunction(X);
        double[] preds = new double[X.rows()];
        for (int i = 0; i < X.rows(); i++) {
            int bestIdx = 0;
            for (int j = 1; j < nClasses; j++) {
                if (scores.get(i, j) > scores.get(i, bestIdx)) {
                    bestIdx = j;
                }
            }
            preds[i] = classes[bestIdx];
        }
        return new Vector(preds);
    }

    /**
     * Compute confidence scores for each class.
     */
    public Matrix decisionFunction(Matrix X) {
        Validation.checkFitted(fitted, "OneVsRestClassifier");
        Validation.checkMatrix(X, -1);

        Matrix scores = new Matrix(X.rows(), nClasses);
        for (int c = 0; c < nClasses; c++) {
            @SuppressWarnings("unchecked")
            Predictor<Matrix, Vector, Vector> pred =
                (Predictor<Matrix, Vector, Vector>) estimators.get(c);
            Vector classScores = pred.predict(X);
            for (int i = 0; i < X.rows(); i++) {
                scores.set(i, c, classScores.get(i));
            }
        }
        return scores;
    }

    @Override
    public double score(Matrix X, Vector y) {
        return org.sklearn.metrics.ClassificationMetrics.accuracyScore(y, predict(X));
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes", classes);
        params.put("estimators", estimators);
        return params;
    }

    public int[] getClasses() {
        return classes;
    }

    public List<Estimator<Matrix, Vector>> getEstimators() {
        return estimators;
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
