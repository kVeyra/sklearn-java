package org.sklearn.svm;

/**
 * Kernel functions for SVM.
 *
 * <p>Provides linear, polynomial, RBF, and sigmoid kernels.
 */
public final class Kernel {

    public static final int LINEAR = 0;
    public static final int POLY = 1;
    public static final int RBF = 2;
    public static final int SIGMOID = 3;

    private Kernel() {
    }

    public static double compute(int kernelType, double dot, double xSq, double ySq,
                                  double gamma, double coef0, int degree) {
        switch (kernelType) {
            case LINEAR:
                return dot;
            case POLY:
                return Math.pow(gamma * dot + coef0, degree);
            case RBF:
                return Math.exp(-gamma * (xSq + ySq - 2 * dot));
            case SIGMOID:
                return Math.tanh(gamma * dot + coef0);
            default:
                throw new IllegalArgumentException("Unknown kernel type: " + kernelType);
        }
    }
}
