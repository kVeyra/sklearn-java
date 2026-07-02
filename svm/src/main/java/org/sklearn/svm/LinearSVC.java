package org.sklearn.svm;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

public class LinearSVC implements Predictor<Matrix, Vector, Vector> {

    private double C;
    private double tol;
    private int maxIter;
    private boolean fitted;
    private SVC svc;

    public LinearSVC() {
        this(1.0, 1e-4, 1000);
    }

    public LinearSVC(double C, double tol, int maxIter) {
        this.C = C;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    @Override
    public LinearSVC fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        svc = new SVC();
        svc.setC(C).setKernel("linear").setTol(tol).setMaxIter(maxIter);
        svc.fit(X, y);
        fitted = true;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "LinearSVC");
        return svc.predict(X);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "LinearSVC");
        return svc.score(X, y);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("C", C);
        p.put("tol", tol);
        p.put("max_iter", maxIter);
        return Collections.unmodifiableMap(p);
    }
}
