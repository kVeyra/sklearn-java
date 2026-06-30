package org.sklearn.svm;

import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

/**
 * Sequential Minimal Optimization (SMO) solver for C-SVC.
 *
 * <p>Solves the SVM dual QP problem using Platt's SMO algorithm.
 * Package-private; used internally by {@link SVC}.
 */
class SMOSolver {

    private final Matrix xMatrix;
    private final int n;
    private final int m;
    private final double cParam;
    private final double tol;
    private final int maxIter;

    private final int kernelType;
    private final double gamma;
    private final double coef0;
    private final int degree;

    private double[] alpha;
    private double b;
    private double[] errorCache;
    private int iter;

    private double[] xSq;
    private double[] yVals;

    SMOSolver(Matrix xMatrix, double cParam, double tol, int maxIter,
              int kernelType, double gamma, double coef0, int degree) {
        this.xMatrix = xMatrix;
        this.n = xMatrix.rows();
        this.m = xMatrix.cols();
        this.cParam = cParam;
        this.tol = tol;
        this.maxIter = maxIter;
        this.kernelType = kernelType;
        this.gamma = gamma;
        this.coef0 = coef0;
        this.degree = degree;
    }

    void solve(Vector y) {
        this.yVals = new double[n];
        for (int i = 0; i < n; i++) {
            yVals[i] = y.get(i);
        }

        alpha = new double[n];
        errorCache = new double[n];
        b = 0.0;

        if (kernelType == Kernel.RBF || kernelType == Kernel.SIGMOID) {
            xSq = new double[n];
            for (int i = 0; i < n; i++) {
                xSq[i] = xMatrix.row(i).normSquared();
            }
        }

        for (int i = 0; i < n; i++) {
            errorCache[i] = -yVals[i];
        }

        int passes = 0;

        while ((maxIter < 0 || iter < maxIter) && passes < Math.max(n, 10)) {
            iter++;
            int numChanged = 0;

            for (int i = 0; i < n; i++) {
                double ei = errorCache[i];
                double yi = yVals[i];
                double ai = alpha[i];
                double ri = ei * yi;

                if ((ri < -tol && ai < cParam) || (ri > tol && ai > 0)) {
                    int j = selectSecond(i, ei);
                    if (j >= 0 && takeStep(i, j)) {
                        numChanged++;
                    }
                }
            }

            if (numChanged == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }
    }

    private int selectSecond(int i1, double e1) {
        int best = -1;
        double maxDiff = 0.0;

        for (int i = 0; i < n; i++) {
            if (alpha[i] > 0 && alpha[i] < cParam) {
                double diff = Math.abs(e1 - errorCache[i]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    best = i;
                }
            }
        }
        if (best < 0) {
            for (int i = 0; i < n; i++) {
                if (i != i1) {
                    double diff = Math.abs(e1 - errorCache[i]);
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        best = i;
                    }
                }
            }
        }
        if (best < 0) {
            best = (i1 + 1) % n;
        }
        return best;
    }

    private boolean takeStep(int i1, int i2) {
        if (i1 == i2) {
            return false;
        }

        double y1 = yVals[i1];
        double y2 = yVals[i2];
        double a1 = alpha[i1];
        double a2 = alpha[i2];
        double e1 = errorCache[i1];
        double e2 = errorCache[i2];

        double k11 = kernel(i1, i1);
        double k22 = kernel(i2, i2);
        double k12 = kernel(i1, i2);
        double eta = k11 + k22 - 2 * k12;

        double L;
        double H;
        if (y1 != y2) {
            L = Math.max(0, a2 - a1);
            H = Math.min(cParam, cParam + a2 - a1);
        } else {
            L = Math.max(0, a1 + a2 - cParam);
            H = Math.min(cParam, a1 + a2);
        }

        if (Math.abs(L - H) < 1e-12) {
            return false;
        }

        double a2New;
        if (eta > 1e-12) {
            a2New = a2 + y2 * (e1 - e2) / eta;
            a2New = Math.max(L, Math.min(H, a2New));
        } else {
            double fL = computeObjectiveAt(i1, i2, a1, a2, L);
            double fH = computeObjectiveAt(i1, i2, a1, a2, H);
            a2New = fL >= fH ? L : H;
        }

        if (Math.abs(a2New - a2) < 1e-12) {
            return false;
        }

        double a1New = a1 + y1 * y2 * (a2 - a2New);

        double oldB = b;

        double b1 = b - e1
                     - y1 * (a1New - a1) * k11
                     - y2 * (a2New - a2) * k12;
        double b2 = b - e2
                     - y1 * (a1New - a1) * k12
                     - y2 * (a2New - a2) * k22;

        if (a1New > 0 && a1New < cParam) {
            b = b1;
        } else if (a2New > 0 && a2New < cParam) {
            b = b2;
        } else {
            b = (b1 + b2) / 2.0;
        }

        double deltaB = b - oldB;
        double delta1 = y1 * (a1New - a1);
        double delta2 = y2 * (a2New - a2);

        alpha[i1] = a1New;
        alpha[i2] = a2New;

        for (int i = 0; i < n; i++) {
            errorCache[i] += delta1 * kernel(i1, i)
                             + delta2 * kernel(i2, i)
                             + deltaB;
        }

        return true;
    }

    private double computeObjectiveAt(int i1, int i2, double a1, double a2,
                                       double a2Test) {
        double a1Test = a1 + yVals[i1] * yVals[i2] * (a2 - a2Test);
        double k11 = kernel(i1, i1);
        double k22 = kernel(i2, i2);
        double k12 = kernel(i1, i2);
        return a1Test + a2Test
               - 0.5 * (a1Test * a1Test * k11
                        + a2Test * a2Test * k22
                        + 2 * yVals[i1] * yVals[i2] * a1Test * a2Test * k12);
    }

    int getIterations() {
        return iter;
    }

    double getB() {
        return b;
    }

    Vector getAlpha() {
        return new Vector(alpha);
    }

    int[] getSupportVectorIndices() {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-12) {
                count++;
            }
        }
        int[] idx = new int[count];
        int p = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-12) {
                idx[p++] = i;
            }
        }
        return idx;
    }

    double decisionFunction(Matrix xNew, int row) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-12) {
                double kVal = kernelEval(i, xNew, row);
                sum += alpha[i] * yVals[i] * kVal;
            }
        }
        return sum + b;
    }

    private double kernelEval(int i, Matrix xNew, int row) {
        double dot = 0.0;
        for (int j = 0; j < m; j++) {
            dot += xMatrix.get(i, j) * xNew.get(row, j);
        }
        if (kernelType == Kernel.LINEAR) {
            return dot;
        }
        if (kernelType == Kernel.POLY) {
            return Math.pow(gamma * dot + coef0, degree);
        }
        double xiSq = xSq[i];
        double xjSq = xNew.row(row).normSquared();
        if (kernelType == Kernel.RBF) {
            double distSq = xiSq + xjSq - 2 * dot;
            return Math.exp(-gamma * distSq);
        }
        return Math.tanh(gamma * dot + coef0);
    }

    private double kernel(int i, int j) {
        double dot = 0.0;
        for (int k = 0; k < m; k++) {
            dot += xMatrix.get(i, k) * xMatrix.get(j, k);
        }
        return Kernel.compute(kernelType, dot,
                              xSq != null ? xSq[i] : 0,
                              xSq != null ? xSq[j] : 0,
                              gamma, coef0, degree);
    }
}
