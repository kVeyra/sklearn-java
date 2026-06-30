package org.sklearn.naive_bayes;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Gaussian Naive Bayes classifier.
 *
 * <p>Assumes features are normally distributed within each class.
 * Computes per-class mean and variance from training data, then
 * classifies via maximum a posteriori (MAP) estimation using
 * log-probabilities.
 *
 * <p>Mirrors {@code sklearn.naive_bayes.GaussianNB}.
 *
 * <p>Usage:
 * <pre>{@code
 * GaussianNB nb = new GaussianNB();
 * nb.fit(X, y);
 * Vector preds = nb.predict(X_test);
 * }</pre>
 */
public class GaussianNB implements Predictor<Matrix, Vector, Vector> {

    private int[] classes;
    private double[] classPrior;
    private Matrix theta; // mean per class per feature
    private Matrix sigma; // variance per class per feature
    private boolean fitted;
    private int nFeatures;

    /**
     * Create a Gaussian Naive Bayes classifier.
     */
    public GaussianNB() {
    }

    @Override
    public GaussianNB fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) y.get(i));
        }
        this.classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);
        int nClasses = classes.length;

        theta = new Matrix(nClasses, m);
        sigma = new Matrix(nClasses, m);
        classPrior = new double[nClasses];

        int[] counts = new int[nClasses];
        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            counts[c]++;
            for (int j = 0; j < m; j++) {
                theta.set(c, j, theta.get(c, j) + X.get(i, j));
            }
        }
        for (int c = 0; c < nClasses; c++) {
            classPrior[c] = (double) counts[c] / n;
            for (int j = 0; j < m; j++) {
                theta.set(c, j, theta.get(c, j) / counts[c]);
            }
        }
        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            for (int j = 0; j < m; j++) {
                double diff = X.get(i, j) - theta.get(c, j);
                sigma.set(c, j, sigma.get(c, j) + diff * diff);
            }
        }
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < m; j++) {
                double var = sigma.get(c, j) / counts[c];
                sigma.set(c, j, Math.max(var, 1e-9));
            }
        }

        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "GaussianNB");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int nClasses = classes.length;
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            int bestClass = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < nClasses; c++) {
                double logProb = Math.log(classPrior[c]);
                for (int j = 0; j < nFeatures; j++) {
                    double diff = X.get(i, j) - theta.get(c, j);
                    double var = sigma.get(c, j);
                    logProb -= 0.5 * Math.log(2 * Math.PI * var) + diff * diff / (2 * var);
                }
                if (logProb > bestScore) {
                    bestScore = logProb;
                    bestClass = c;
                }
            }
            preds[i] = classes[bestClass];
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "GaussianNB");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
    }

    private int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return i;
            }
        }
        return 0;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("classes_", classes);
        params.put("class_prior_", classPrior);
        params.put("theta_", theta);
        params.put("sigma_", sigma);
        return Collections.unmodifiableMap(params);
    }
}
