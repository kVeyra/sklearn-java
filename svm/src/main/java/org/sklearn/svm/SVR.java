package org.sklearn.svm;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Epsilon-Support Vector Regression.
 *
 * <p>Fits a support vector machine regressor using an SMO-style algorithm.
 * Supports linear, polynomial, RBF, and sigmoid kernels.
 *
 * <p>Mirrors {@code sklearn.svm.SVR}.
 */
public class SVR implements Predictor<Matrix, Vector, Vector> {

    private double cParam;
    private double epsilon;
    private String kernel;
    private int degree;
    private String gammaMode;
    private double gammaValue;
    private double coef0;
    private double tol;
    private int maxIter;
    private boolean fitted;

    private int nSamples;
    private int nFeatures;
    private Matrix xTrain;
    private Vector yTrain;

    private double[] beta;
    private double b;
    private int[] supportIndices;

    private static final Map<String, Integer> KERNEL_MAP = Map.of(
        "linear", Kernel.LINEAR,
        "poly", Kernel.POLY,
        "rbf", Kernel.RBF,
        "sigmoid", Kernel.SIGMOID
    );

    public SVR() {
        this.cParam = 1.0;
        this.epsilon = 0.1;
        this.kernel = "rbf";
        this.degree = 3;
        this.gammaMode = "scale";
        this.gammaValue = 0.0;
        this.coef0 = 0.0;
        this.tol = 1e-3;
        this.maxIter = -1;
    }

    public SVR setC(double c) {
        this.cParam = c;
        return this;
    }

    public SVR setEpsilon(double eps) {
        this.epsilon = eps;
        return this;
    }

    public SVR setKernel(String k) {
        if (!KERNEL_MAP.containsKey(k)) {
            throw new IllegalArgumentException("Unknown kernel: " + k);
        }
        this.kernel = k;
        return this;
    }

    public SVR setDegree(int d) {
        this.degree = d;
        return this;
    }

    public SVR setGamma(String g) {
        this.gammaMode = g;
        this.gammaValue = 0.0;
        return this;
    }

    public SVR setGamma(double g) {
        this.gammaMode = "value";
        this.gammaValue = g;
        return this;
    }

    public SVR setCoef0(double c) {
        this.coef0 = c;
        return this;
    }

    public SVR setTol(double t) {
        this.tol = t;
        return this;
    }

    public SVR setMaxIter(int mi) {
        this.maxIter = mi;
        return this;
    }

    private int resolveKernelType() {
        return KERNEL_MAP.getOrDefault(kernel, Kernel.RBF);
    }

    private double resolveGamma(Matrix x) {
        if (gammaMode.equals("value")) {
            return gammaValue;
        }
        int nf = x.cols();
        if (gammaMode.equals("scale")) {
            double var = 0.0;
            for (int j = 0; j < nf; j++) {
                double mean = 0.0;
                for (int i = 0; i < x.rows(); i++) {
                    mean += x.get(i, j);
                }
                mean /= x.rows();
                double colVar = 0.0;
                for (int i = 0; i < x.rows(); i++) {
                    double diff = x.get(i, j) - mean;
                    colVar += diff * diff;
                }
                colVar /= x.rows();
                var += colVar;
            }
            var /= nf;
            return var > 0 ? 1.0 / (nf * var) : 1.0 / nf;
        }
        return 1.0 / nf;
    }

    @Override
    public SVR fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());
        this.nSamples = X.rows();
        this.nFeatures = X.cols();
        this.xTrain = new Matrix(X);
        this.yTrain = new Vector(y);
        fitSVR(resolveGamma(X), resolveKernelType());
        this.fitted = true;
        return this;
    }

    private void fitSVR(double gamma, int kt) {
        int n = nSamples;
        beta = new double[n];
        b = 0.0;

        double[] e = new double[n];
        for (int i = 0; i < n; i++) {
            e[i] = -yTrain.get(i);
        }

        int maxPasses = Math.max(100, n * 5);
        int passes = 0;
        int iter = 0;

        while ((maxIter < 0 || iter < maxIter) && passes < maxPasses) {
            iter++;
            int changed = 0;
            for (int i = 0; i < n; i++) {
                if (kktViolation(i, e[i])) {
                    int j = findPartner(i, e);
                    if (j >= 0 && updateBeta(i, j, e, gamma, kt)) {
                        changed++;
                    }
                }
            }
            if (changed == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }

        computeBiasFromBound(e);

        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(beta[i]) > 1e-12) {
                cnt++;
            }
        }
        supportIndices = new int[cnt];
        int p = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(beta[i]) > 1e-12) {
                supportIndices[p++] = i;
            }
        }
    }

    private boolean kktViolation(int i, double ei) {
        double bi = beta[i];
        double absEi = Math.abs(ei);
        if (bi == 0) {
            return absEi > epsilon + tol;
        }
        if (bi > 0 && bi < cParam) {
            return Math.abs(ei - epsilon) > tol;
        }
        if (bi < 0 && bi > -cParam) {
            return Math.abs(ei + epsilon) > tol;
        }
        if (bi >= cParam) {
            return ei < epsilon - tol;
        }
        return ei > -epsilon + tol;
    }

    private int findPartner(int i, double[] e) {
        int best = -1;
        double max = 0;
        for (int j = 0; j < nSamples; j++) {
            if (j != i) {
                double d = Math.abs(e[i] - e[j]);
                if (d > max) {
                    max = d;
                    best = j;
                }
            }
        }
        return best;
    }

    private boolean updateBeta(int i, int j, double[] e, double gamma, int kt) {
        if (i == j) {
            return false;
        }

        double kii = kernelVal(i, i, gamma, kt);
        double kjj = kernelVal(j, j, gamma, kt);
        double kij = kernelVal(i, j, gamma, kt);
        double eta = kii + kjj - 2 * kij;

        if (eta < 1e-12) {
            return false;
        }

        double oldB = b;
        double bi = beta[i];
        double bj = beta[j];

        double newBj = bj + (e[i] - e[j]) / eta;
        newBj = Math.max(-cParam, Math.min(cParam, newBj));
        double newBi = bi + (bj - newBj);

        if (Math.abs(newBi - bi) < 1e-12 && Math.abs(newBj - bj) < 1e-12) {
            return false;
        }

        double di = newBi - bi;
        double dj = newBj - bj;

        double b1 = b - e[i] - di * kii - dj * kij;
        double b2 = b - e[j] - di * kij - dj * kjj;

        if (Math.abs(newBi) > 0 && Math.abs(newBi) < cParam) {
            b = b1;
        } else if (Math.abs(newBj) > 0 && Math.abs(newBj) < cParam) {
            b = b2;
        } else {
            b = (b1 + b2) / 2.0;
        }

        beta[i] = newBi;
        beta[j] = newBj;

        for (int k = 0; k < nSamples; k++) {
            e[k] += di * kernelVal(k, i, gamma, kt)
                    + dj * kernelVal(k, j, gamma, kt)
                    + (b - oldB);
        }

        return true;
    }

    private void computeBiasFromBound(double[] e) {
        int cnt = 0;
        double sum = 0;
        for (int i = 0; i < nSamples; i++) {
            if (Math.abs(beta[i]) > 0 && Math.abs(beta[i]) < cParam) {
                double target = beta[i] > 0 ? epsilon : -epsilon;
                sum += (yTrain.get(i) + target) - (e[i] + yTrain.get(i) - b - target);
                cnt++;
            }
        }
        if (cnt > 0) {
            b = sum / cnt;
        }
    }

    private double kernelVal(int i, int j, double gamma, int kt) {
        double dot = 0.0;
        for (int k = 0; k < nFeatures; k++) {
            dot += xTrain.get(i, k) * xTrain.get(j, k);
        }
        if (kt == Kernel.LINEAR) {
            return dot;
        }
        double xiSq = 0;
        double xjSq = 0;
        if (kt == Kernel.RBF || kt == Kernel.SIGMOID) {
            for (int k = 0; k < nFeatures; k++) {
                xiSq += xTrain.get(i, k) * xTrain.get(i, k);
                xjSq += xTrain.get(j, k) * xTrain.get(j, k);
            }
        }
        return Kernel.compute(kt, dot, xiSq, xjSq, gamma, coef0, degree);
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "SVR");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Feature dimension mismatch");
        }
        int n = X.rows();
        double[] preds = new double[n];
        double gamma = resolveGamma(xTrain);
        int kt = resolveKernelType();
        for (int r = 0; r < n; r++) {
            double sum = 0.0;
            for (int si : supportIndices) {
                double dot = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    dot += xTrain.get(si, j) * X.get(r, j);
                }
                double kVal;
                if (kt == Kernel.LINEAR) {
                    kVal = dot;
                } else {
                    double xiSq = xTrain.row(si).normSquared();
                    double xjSq = X.row(r).normSquared();
                    kVal = Kernel.compute(kt, dot, xiSq, xjSq, gamma, coef0, degree);
                }
                sum += beta[si] * kVal;
            }
            preds[r] = sum + b;
        }
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "SVR");
        Vector pred = predict(X);
        double ssRes = 0;
        double ssTot = 0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double d = y.get(i) - pred.get(i);
            ssRes += d * d;
            double dm = y.get(i) - yMean;
            ssTot += dm * dm;
        }
        if (ssTot == 0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("C", cParam);
        p.put("epsilon", epsilon);
        p.put("kernel", kernel);
        p.put("degree", degree);
        p.put("gamma", gammaMode.equals("value") ? gammaValue : gammaMode);
        p.put("coef0", coef0);
        p.put("tol", tol);
        p.put("max_iter", maxIter);
        return Collections.unmodifiableMap(p);
    }
}
