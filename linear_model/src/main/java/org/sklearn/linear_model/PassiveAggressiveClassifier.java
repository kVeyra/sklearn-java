package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Passive Aggressive Classifier.
 */
public class PassiveAggressiveClassifier implements Predictor<Matrix, Vector, Vector> {

    private String loss;
    private double c;
    private boolean fitIntercept;
    private int maxIter;
    private double tol;
    private boolean shuffle;
    private long randomState;

    private SGDClassifier sgd;
    private boolean fitted;

    public PassiveAggressiveClassifier() {
        this("hinge", 1.0, true, 1000, 1e-3, true, 42);
    }

    public PassiveAggressiveClassifier(String loss, double C, boolean fitIntercept,
                                        int maxIter, double tol, boolean shuffle, long randomState) {
        this.loss = loss;
        this.c = C;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.shuffle = shuffle;
        this.randomState = randomState;
        buildSGD();
    }

    private void buildSGD() {
        String lr = loss.equals("hinge") ? "pa1" : "pa2";
        String sgdLoss = loss;
        this.sgd = new SGDClassifier(sgdLoss, "l2", 1.0, 0.15,
            fitIntercept, maxIter, tol, lr, c, 0.5, false, 0.1, 5,
            shuffle, randomState, false, false);
    }

    @Override
    public PassiveAggressiveClassifier fit(Matrix X, Vector y) {
        sgd.fit(X, y);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "PassiveAggressiveClassifier");
        return sgd.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "PassiveAggressiveClassifier");
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
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss); p.put("C", c); p.put("fit_intercept", fitIntercept);
        p.put("max_iter", maxIter); p.put("tol", tol);
        return Collections.unmodifiableMap(p);
    }
}
