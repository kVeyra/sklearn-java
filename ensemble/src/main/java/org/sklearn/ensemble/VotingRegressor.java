package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Voting regressor.
 *
 * <p>Averages predictions from multiple regressors.
 *
 * <p>Mirrors {@code sklearn.ensemble.VotingRegressor}.
 */
public class VotingRegressor implements Predictor<Matrix, Vector, Vector> {

    private List<Estimator<Matrix, Vector>> estimators;
    private List<Predictor<Matrix, Vector, Vector>> fittedEstimators;

    public VotingRegressor(List<Estimator<Matrix, Vector>> estimators) {
        this.estimators = estimators;
    }

    @Override
    @SuppressWarnings("unchecked")
    public VotingRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        fittedEstimators = new ArrayList<>();
        for (Estimator<Matrix, Vector> est : estimators) {
            Estimator<Matrix, Vector> clone = cloneEstimator(est);
            clone.fit(X, y);
            if (clone instanceof Predictor) {
                fittedEstimators.add((Predictor<Matrix, Vector, Vector>) clone);
            }
        }
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        int n = X.rows();
        double[] sum = new double[n];

        for (Predictor<Matrix, Vector, Vector> est : fittedEstimators) {
            Vector pred = est.predict(X);
            for (int i = 0; i < n; i++) {
                sum[i] += pred.get(i);
            }
        }

        Vector result = new Vector(n);
        int nEst = fittedEstimators.size();
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
        p.put("n_estimators", estimators.size());
        return p;
    }

    @SuppressWarnings("unchecked")
    private Estimator<Matrix, Vector> cloneEstimator(Estimator<Matrix, Vector> est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator", e);
        }
    }
}
