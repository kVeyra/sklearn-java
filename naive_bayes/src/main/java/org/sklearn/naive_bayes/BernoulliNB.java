package org.sklearn.naive_bayes;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Naive Bayes classifier for binary/boolean features.
 */
public class BernoulliNB implements Predictor<Matrix, Vector, Vector> {

    private double alpha;
    private boolean fitPrior;
    private double[] classPrior;
    private boolean binarize;
    private double threshold;
    private boolean fitted;

    private int[] classes;
    private double[] classLogPrior;
    private Matrix featureLogProb;
    private Matrix featureCount;
    private double[] classCount;
    private Matrix negFeatureLogProb;
    private int nFeatures;

    public BernoulliNB() {
        this(1.0, true, null, true, 0.0);
    }

    public BernoulliNB(double alpha, boolean fitPrior, double[] classPrior,
                       boolean binarize, double threshold) {
        this.alpha = alpha;
        this.fitPrior = fitPrior;
        this.classPrior = classPrior;
        this.binarize = binarize;
        this.threshold = threshold;
    }

    @Override
    public BernoulliNB fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        if (binarize) {
            X = binarizeMatrix(X);
        }

        Set<Integer> uniqueLabels = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) {
            uniqueLabels.add((int) y.get(i));
        }
        this.classes = uniqueLabels.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(this.classes);
        int nClasses = classes.length;

        classCount = new double[nClasses];
        featureCount = new Matrix(nClasses, m);

        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            classCount[c]++;
            for (int j = 0; j < m; j++) {
                if (X.get(i, j) > 0) {
                    featureCount.set(c, j, featureCount.get(c, j) + 1);
                }
            }
        }

        updateClassLogPrior();
        updateFeatureLogProb();

        this.fitted = true;
        return this;
    }

    private void updateClassLogPrior() {
        int nClasses = classes.length;
        classLogPrior = new double[nClasses];
        if (classPrior != null) {
            if (classPrior.length != nClasses) {
                throw new IllegalArgumentException("Number of priors must match number of classes.");
            }
            double sum = 0.0;
            for (double p : classPrior) {
                sum += p;
            }
            if (Math.abs(sum - 1.0) > 1e-10) {
                throw new IllegalArgumentException("Priors must sum to 1.");
            }
            for (int c = 0; c < nClasses; c++) {
                classLogPrior[c] = Math.log(classPrior[c]);
            }
        } else if (fitPrior) {
            double total = 0.0;
            for (double c : classCount) {
                total += c;
            }
            for (int c = 0; c < nClasses; c++) {
                classLogPrior[c] = Math.log(classCount[c] / total);
            }
        } else {
            double logP = -Math.log(nClasses);
            Arrays.fill(classLogPrior, logP);
        }
    }

    private Matrix binarizeMatrix(Matrix X) {
        Matrix result = new Matrix(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                result.set(i, j, X.get(i, j) > threshold ? 1.0 : 0.0);
            }
        }
        return result;
    }

    private void updateFeatureLogProb() {
        int nClasses = classes.length;
        int m = nFeatures;
        featureLogProb = new Matrix(nClasses, m);
        negFeatureLogProb = new Matrix(nClasses, m);

        for (int c = 0; c < nClasses; c++) {
            double total = classCount[c];
            double smoothedTotal = total + 2.0 * alpha;
            for (int j = 0; j < m; j++) {
                double count = featureCount.get(c, j);
                double prob = (count + alpha) / smoothedTotal;
                featureLogProb.set(c, j, Math.log(prob));
                negFeatureLogProb.set(c, j, Math.log(1.0 - prob));
            }
        }
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "BernoulliNB");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        if (binarize) {
            X = binarizeMatrix(X);
        }

        int n = X.rows();
        int nClasses = classes.length;
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            int bestClass = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < nClasses; c++) {
                double score = classLogPrior[c];
                for (int j = 0; j < nFeatures; j++) {
                    if (X.get(i, j) > 0) {
                        score += featureLogProb.get(c, j);
                    } else {
                        score += negFeatureLogProb.get(c, j);
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            preds[i] = classes[bestClass];
        }
        return new Vector(preds);
    }

    public Vector predictLogProbasVector(Matrix X) {
        Validation.checkFitted(fitted, "BernoulliNB");
        double[] logProbas = predictLogProbasInternal(X);
        return new Vector(logProbas);
    }

    public double[] predictProbas(Matrix X) {
        double[] logProbas = predictLogProbasInternal(X);
        double[] probas = new double[logProbas.length];
        for (int i = 0; i < logProbas.length; i++) {
            probas[i] = Math.exp(logProbas[i]);
        }
        return probas;
    }

    private double[] predictLogProbasInternal(Matrix X) {
        int n = X.rows();
        int nClasses = classes.length;
        double[][] jll = jointLogLikelihood(X);
        double[] result = new double[n * nClasses];

        for (int i = 0; i < n; i++) {
            double maxVal = jll[i][0];
            for (int c = 1; c < nClasses; c++) {
                if (jll[i][c] > maxVal) {
                    maxVal = jll[i][c];
                }
            }
            double sumExp = 0.0;
            for (int c = 0; c < nClasses; c++) {
                sumExp += Math.exp(jll[i][c] - maxVal);
            }
            double logSum = maxVal + Math.log(sumExp);
            for (int c = 0; c < nClasses; c++) {
                result[i * nClasses + c] = jll[i][c] - logSum;
            }
        }
        return result;
    }

    private double[][] jointLogLikelihood(Matrix X) {
        Matrix xb = binarize ? binarizeMatrix(X) : X;
        int n = xb.rows();
        int nClasses = classes.length;
        double[][] jll = new double[n][nClasses];
        for (int c = 0; c < nClasses; c++) {
            for (int i = 0; i < n; i++) {
                double score = classLogPrior[c];
                for (int j = 0; j < nFeatures; j++) {
                    if (xb.get(i, j) > 0) {
                        score += featureLogProb.get(c, j);
                    } else {
                        score += negFeatureLogProb.get(c, j);
                    }
                }
                jll[i][c] = score;
            }
        }
        return jll;
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

    public int[] getClasses() {
        return classes;
    }
    public double[] getClassLogPrior() {
        return classLogPrior;
    }
    public Matrix getFeatureLogProb() {
        return featureLogProb;
    }
    public Matrix getNegFeatureLogProb() {
        return negFeatureLogProb;
    }
    public Matrix getFeatureCount() {
        return featureCount;
    }
    public double[] getClassCount() {
        return classCount;
    }
    public boolean isFitted() {
        return fitted;
    }

    private int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alpha", alpha);
        p.put("fit_prior", fitPrior);
        p.put("class_prior", classPrior);
        p.put("binarize", binarize);
        p.put("threshold", threshold);
        return Collections.unmodifiableMap(p);
    }
}
