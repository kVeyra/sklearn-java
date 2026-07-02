package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Perceptron classifier.
 */
public class Perceptron implements Predictor<Matrix, Vector, Vector> {

    private SGDClassifier sgd;
    private boolean fitted;

    public Perceptron() {
        this(null, 0.0001, true, 1000, 1e-3, true, 42);
    }

    public Perceptron(String penalty, double alpha, boolean fitIntercept,
                       int maxIter, double tol, boolean shuffle, long randomState) {
        this.sgd = new SGDClassifier("perceptron",
            penalty != null ? penalty : "l2",
            alpha, 0.15, fitIntercept, maxIter, tol,
            "constant", 1.0, 0.5, false, 0.1, 5, shuffle, randomState, false, false);
    }

    @Override
    public Perceptron fit(Matrix X, Vector y) {
        sgd.fit(X, y);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "Perceptron");
        return sgd.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "Perceptron");
        return sgd.score(X, y);
    }

    public Vector[] getCoefs() {
        return sgd.getCoefs();
    }
    public Vector getIntercepts() {
        return sgd.getIntercepts();
    }
    public int[] getClasses() {
        return sgd.getClasses();
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        return sgd.getParameters();
    }
}
