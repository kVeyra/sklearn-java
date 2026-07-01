package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Voting classifier (hard/soft voting).
 *
 * <p>Aggregates predictions from multiple classifiers.
 * Hard voting: majority vote. Soft voting: average of predicted probabilities.
 *
 * <p>Mirrors {@code sklearn.ensemble.VotingClassifier}.
 */
public class VotingClassifier implements Predictor<Matrix, Vector, Vector> {

    private List<Estimator<Matrix, Vector>> estimators;
    private String voting;
    private List<Predictor<Matrix, Vector, Vector>> fittedEstimators;
    private int[] classes;

    public VotingClassifier(List<Estimator<Matrix, Vector>> estimators, String voting) {
        this.estimators = estimators;
        this.voting = voting;
    }

    @Override
    @SuppressWarnings("unchecked")
    public VotingClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Set<Integer> clsSet = new TreeSet<>();
        for (int i = 0; i < y.size(); i++) {
            clsSet.add((int) y.get(i));
        }
        classes = new int[clsSet.size()];
        int ci = 0;
        for (int c : clsSet) {
            classes[ci++] = c;
        }

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
        int nEst = fittedEstimators.size();

        if ("hard".equals(voting)) {
            int[][] votes = new int[n][classes.length];
            for (Predictor<Matrix, Vector, Vector> est : fittedEstimators) {
                Vector pred = est.predict(X);
                for (int i = 0; i < n; i++) {
                    int pClass = (int) pred.get(i);
                    for (int k = 0; k < classes.length; k++) {
                        if (Math.abs(classes[k] - pClass) < 0.5) {
                            votes[i][k]++;
                            break;
                        }
                    }
                }
            }
            Vector result = new Vector(n);
            for (int i = 0; i < n; i++) {
                int bestK = 0;
                for (int k = 1; k < classes.length; k++) {
                    if (votes[i][k] > votes[i][bestK]) {
                        bestK = k;
                    }
                }
                result.set(i, classes[bestK]);
            }
            return result;
        } else {
            double[][] probSums = new double[n][classes.length];
            for (Predictor<Matrix, Vector, Vector> est : fittedEstimators) {
                // Try predictProba first
                try {
                    java.lang.reflect.Method m = est.getClass().getMethod("predictProba", Matrix.class);
                    Vector prob = (Vector) m.invoke(est, X);
                    for (int i = 0; i < n; i++) {
                        double p = prob.get(i);
                        if (classes.length == 2) {
                            probSums[i][1] += p;
                            probSums[i][0] += 1.0 - p;
                        } else {
                            for (int k = 0; k < classes.length; k++) {
                                probSums[i][k] += prob.get(i * classes.length + k);
                            }
                        }
                    }
                } catch (Exception e) {
                    Vector pred = est.predict(X);
                    for (int i = 0; i < n; i++) {
                        int pClass = (int) pred.get(i);
                        for (int k = 0; k < classes.length; k++) {
                            if (Math.abs(classes[k] - pClass) < 0.5) {
                                probSums[i][k] += 1.0;
                                break;
                            }
                        }
                    }
                }
            }
            Vector result = new Vector(n);
            for (int i = 0; i < n; i++) {
                int bestK = 0;
                for (int k = 1; k < classes.length; k++) {
                    if (probSums[i][k] > probSums[i][bestK]) {
                        bestK = k;
                    }
                }
                result.set(i, classes[bestK]);
            }
            return result;
        }
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("voting", voting);
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
