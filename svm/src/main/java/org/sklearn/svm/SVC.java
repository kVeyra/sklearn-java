package org.sklearn.svm;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * C-Support Vector Classification.
 *
 * <p>Fits a support vector machine classifier using the SMO algorithm.
 * Supports linear, poly, RBF, and sigmoid kernels.
 * Multi-class uses one-vs-one (like sklearn's SVC).
 *
 * <p>Mirrors {@code sklearn.svm.SVC}.
 */
public class SVC implements Predictor<Matrix, Vector, Vector> {

    private double cParam;
    private String kernel;
    private int degree;
    private String gammaMode;
    private double gammaValue;
    private double coef0;
    private double tol;
    private int maxIter;
    private boolean fitted;

    private int[] classes;
    private int nClasses;

    private double b;
    private Vector alphaVec;
    private int[] supportIndices;
    private double[] binaryYLabels;

    private List<BinaryModel> binaryModels;

    private int nFeatures;
    private Matrix xTrain;

    private static final Map<String, Integer> KERNEL_MAP = Map.of(
        "linear", Kernel.LINEAR,
        "poly", Kernel.POLY,
        "rbf", Kernel.RBF,
        "sigmoid", Kernel.SIGMOID
    );

    private static class BinaryModel {
        final int classA;
        final int classB;
        final double b;
        final double[] alpha;
        final double[] yVals;

        BinaryModel(int classA, int classB, double b, double[] alpha,
                    double[] yVals) {
            this.classA = classA;
            this.classB = classB;
            this.b = b;
            this.alpha = alpha;
            this.yVals = yVals;
        }
    }

    public SVC() {
        this.cParam = 1.0;
        this.kernel = "rbf";
        this.degree = 3;
        this.gammaMode = "scale";
        this.gammaValue = 0.0;
        this.coef0 = 0.0;
        this.tol = 1e-3;
        this.maxIter = -1;
    }

    public SVC setC(double c) {
        this.cParam = c;
        return this;
    }

    public SVC setKernel(String k) {
        if (!KERNEL_MAP.containsKey(k)) {
            throw new IllegalArgumentException(
                "Unknown kernel: " + k + ". Supported: linear, poly, rbf, sigmoid");
        }
        this.kernel = k;
        return this;
    }

    public SVC setDegree(int d) {
        this.degree = d;
        return this;
    }

    public SVC setGamma(String g) {
        this.gammaMode = g;
        this.gammaValue = 0.0;
        return this;
    }

    public SVC setGamma(double g) {
        this.gammaMode = "value";
        this.gammaValue = g;
        return this;
    }

    public SVC setCoef0(double c) {
        this.coef0 = c;
        return this;
    }

    public SVC setTol(double t) {
        this.tol = t;
        return this;
    }

    public SVC setMaxIter(int mi) {
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
    public SVC fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        this.nFeatures = X.cols();
        this.xTrain = new Matrix(X);

        Set<Integer> labels = new LinkedHashSet<>();
        for (int i = 0; i < y.size(); i++) {
            labels.add((int) y.get(i));
        }
        this.classes = labels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);
        this.nClasses = classes.length;

        int effectiveMaxIter = maxIter > 0 ? maxIter : Integer.MAX_VALUE;
        double gamma = resolveGamma(X);
        int kt = resolveKernelType();

        if (nClasses == 2) {
            fitBinary(X, y, gamma, kt, effectiveMaxIter);
        } else {
            fitMultiClass(X, y, gamma, kt, effectiveMaxIter);
        }

        this.fitted = true;
        return this;
    }

    private void fitBinary(Matrix X, Vector y, double gamma, int kt, int effMax) {
        Vector binaryY = new Vector(X.rows());
        for (int i = 0; i < X.rows(); i++) {
            binaryY.set(i, y.get(i) == classes[0] ? 1.0 : -1.0);
        }

        SMOSolver solver = new SMOSolver(X, cParam, tol,
            effMax == Integer.MAX_VALUE ? -1 : effMax, kt, gamma, coef0, degree);
        solver.solve(binaryY);

        this.b = solver.getB();
        this.alphaVec = solver.getAlpha();
        this.supportIndices = solver.getSupportVectorIndices();
        this.binaryYLabels = new double[X.rows()];
        for (int i = 0; i < X.rows(); i++) {
            binaryYLabels[i] = binaryY.get(i);
        }
    }

    private void fitMultiClass(Matrix X, Vector y, double gamma, int kt, int effMax) {
        binaryModels = new ArrayList<>();

        for (int ci = 0; ci < nClasses; ci++) {
            for (int cj = ci + 1; cj < nClasses; cj++) {
                int classA = classes[ci];
                int classB = classes[cj];

                int count = 0;
                for (int i = 0; i < X.rows(); i++) {
                    int label = (int) y.get(i);
                    if (label == classA || label == classB) {
                        count++;
                    }
                }

                Matrix subX = new Matrix(count, nFeatures);
                double[] subY = new double[count];
                int[] subToFull = new int[count];
                int idx = 0;
                for (int i = 0; i < X.rows(); i++) {
                    int label = (int) y.get(i);
                    if (label == classA || label == classB) {
                        for (int j = 0; j < nFeatures; j++) {
                            subX.set(idx, j, X.get(i, j));
                        }
                        subY[idx] = label == classA ? 1.0 : -1.0;
                        subToFull[idx] = i;
                        idx++;
                    }
                }

                Vector subYVec = new Vector(subY);
                SMOSolver solver = new SMOSolver(subX, cParam, tol,
                    effMax == Integer.MAX_VALUE ? -1 : effMax, kt, gamma, coef0, degree);
                solver.solve(subYVec);

                double[] fullAlpha = new double[X.rows()];
                double[] fullY = new double[X.rows()];
                int[] svIndices = solver.getSupportVectorIndices();
                Vector alphVec = solver.getAlpha();
                for (int si : svIndices) {
                    fullAlpha[subToFull[si]] = alphVec.get(si);
                }
                for (int k = 0; k < count; k++) {
                    fullY[subToFull[k]] = subY[k];
                }

                binaryModels.add(new BinaryModel(
                    classA, classB, solver.getB(), fullAlpha, fullY));
            }
        }
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "SVC");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];
        double gamma = resolveGamma(xTrain);
        int kt = resolveKernelType();

        if (nClasses == 2) {
            for (int i = 0; i < n; i++) {
                double f = decisionFunctionBinary(X, i, gamma, kt);
                preds[i] = f >= 0 ? classes[0] : classes[1];
            }
        } else {
            int[][] votes = new int[n][nClasses];
            for (BinaryModel model : binaryModels) {
                int idxA = indexOfClass(model.classA);
                int idxB = indexOfClass(model.classB);
                for (int i = 0; i < n; i++) {
                    double f = decisionFunctionMulti(X, i, model, gamma, kt);
                    if (f >= 0) {
                        votes[i][idxA]++;
                    } else {
                        votes[i][idxB]++;
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                int best = 0;
                int bestVotes = -1;
                for (int c = 0; c < nClasses; c++) {
                    if (votes[i][c] > bestVotes) {
                        bestVotes = votes[i][c];
                        best = c;
                    }
                }
                preds[i] = classes[best];
            }
        }

        return new Vector(preds);
    }

    private double decisionFunctionBinary(Matrix xNew, int row, double gamma, int kt) {
        double sum = 0.0;
        for (int si : supportIndices) {
            if (alphaVec.get(si) > 1e-12) {
                double kVal = kernelEval(xTrain, si, xNew, row, gamma, kt);
                sum += alphaVec.get(si) * binaryYLabels[si] * kVal;
            }
        }
        return sum + b;
    }

    private double decisionFunctionMulti(Matrix xNew, int row, BinaryModel model,
                                          double gamma, int kt) {
        double sum = 0.0;
        for (int i = 0; i < model.alpha.length; i++) {
            if (model.alpha[i] > 1e-12) {
                double kVal = kernelEval(xTrain, i, xNew, row, gamma, kt);
                sum += model.alpha[i] * model.yVals[i] * kVal;
            }
        }
        return sum + model.b;
    }

    private double kernelEval(Matrix xTr, int svIdx, Matrix xNew, int row,
                               double gamma, int kt) {
        double dot = 0.0;
        for (int j = 0; j < nFeatures; j++) {
            dot += xTr.get(svIdx, j) * xNew.get(row, j);
        }
        if (kt == Kernel.LINEAR) {
            return dot;
        }
        if (kt == Kernel.POLY) {
            return Math.pow(gamma * dot + coef0, degree);
        }
        double xiSq = xTr.row(svIdx).normSquared();
        double xjSq = xNew.row(row).normSquared();
        if (kt == Kernel.RBF) {
            double distSq = xiSq + xjSq - 2 * dot;
            return Math.exp(-gamma * distSq);
        }
        return Math.tanh(gamma * dot + coef0);
    }

    private int indexOfClass(int c) {
        for (int i = 0; i < classes.length; i++) {
            if (classes[i] == c) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "SVC");
        Vector pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (Math.abs(pred.get(i) - y.get(i)) < 0.5) {
                correct++;
            }
        }
        return (double) correct / y.size();
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("C", cParam);
        params.put("kernel", kernel);
        params.put("degree", degree);
        params.put("gamma", gammaMode.equals("value") ? gammaValue : gammaMode);
        params.put("coef0", coef0);
        params.put("tol", tol);
        params.put("max_iter", maxIter);
        return Collections.unmodifiableMap(params);
    }
}
