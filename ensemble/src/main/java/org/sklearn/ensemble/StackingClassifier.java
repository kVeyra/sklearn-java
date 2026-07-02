package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Stacking classifier.
 *
 * <p>Stacks multiple classifiers using a final meta-classifier.
 * Base estimators are fitted on the full training data, and their
 * predictions (or probabilities) are used as features for the
 * final estimator via cross-validation.
 *
 * <p>Mirrors {@code sklearn.ensemble.StackingClassifier}.
 */
public class StackingClassifier implements Predictor<Matrix, Vector, Vector> {

    private List<Estimator<Matrix, Vector>> estimators;
    private Estimator<Matrix, Vector> finalEstimator;
    private int cv;
    private List<Predictor<Matrix, Vector, Vector>> fittedEstimators;
    private Predictor<Matrix, Vector, Vector> fittedFinal;
    private int[] classes;

    public StackingClassifier(List<Estimator<Matrix, Vector>> estimators,
                               Estimator<Matrix, Vector> finalEstimator, int cv) {
        this.estimators = estimators;
        this.finalEstimator = finalEstimator;
        this.cv = cv;
    }

    @Override
    @SuppressWarnings("unchecked")
    public StackingClassifier fit(Matrix X, Vector y) {
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

        int n = X.rows();

        // Fit base estimators on full data
        fittedEstimators = new ArrayList<>();
        for (Estimator<Matrix, Vector> est : estimators) {
            Estimator<Matrix, Vector> clone = cloneEstimator(est);
            clone.fit(X, y);
            if (clone instanceof Predictor) {
                fittedEstimators.add((Predictor<Matrix, Vector, Vector>) clone);
            }
        }

        // Generate meta-features via cross-validation
        int nBase = fittedEstimators.size();
        int nClassesVal = classes.length;
        int nMetaFeatures = nBase * nClassesVal;
        double[][] metaFeatures = new double[n][nMetaFeatures];

        // Simple hold-out CV: split into cv folds
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(42));

        int foldSize = n / cv;
        for (int fold = 0; fold < cv; fold++) {
            int testStart = fold * foldSize;
            int testEnd = (fold == cv - 1) ? n : testStart + foldSize;
            Set<Integer> testSet = new HashSet<>(indices.subList(testStart, testEnd));

            List<Integer> trainIdx = new ArrayList<>();
            List<Integer> testIdx = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (testSet.contains(indices.get(i))) {
                    testIdx.add(indices.get(i));
                } else {
                    trainIdx.add(indices.get(i));
                }
            }

            Matrix xTrain = extractRows(X, trainIdx);
            Vector yTrain = extractRows(y, trainIdx);
            Matrix xTest = extractRows(X, testIdx);

            for (int b = 0; b < nBase; b++) {
                Estimator<Matrix, Vector> clone = cloneEstimator(fittedEstimators.get(b));
                clone.fit(xTrain, yTrain);
                if (clone instanceof Predictor) {
                    Predictor<Matrix, Vector, Vector> predClone =
                        (Predictor<Matrix, Vector, Vector>) clone;
                    Vector pred = predClone.predict(xTest);
                    for (int i = 0; i < testIdx.size(); i++) {
                        int row = testIdx.get(i);
                        int pClass = (int) pred.get(i);
                        for (int k = 0; k < nClassesVal; k++) {
                            metaFeatures[row][b * nClassesVal + k] =
                                Math.abs(classes[k] - pClass) < 0.5 ? 1.0 : 0.0;
                        }
                    }
                }
            }
        }

        // Fit final estimator on meta-features
        Estimator<Matrix, Vector> finalClone = cloneEstimator(finalEstimator);
        finalClone.fit(new Matrix(metaFeatures), y);
        fittedFinal = (Predictor<Matrix, Vector, Vector>) finalClone;
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        int n = X.rows();
        int nBase = fittedEstimators.size();
        int nClassesVal = classes.length;
        double[][] metaFeatures = new double[n][nBase * nClassesVal];

        for (int b = 0; b < nBase; b++) {
            Vector pred = fittedEstimators.get(b).predict(X);
            for (int i = 0; i < n; i++) {
                int pClass = (int) pred.get(i);
                for (int k = 0; k < nClassesVal; k++) {
                    metaFeatures[i][b * nClassesVal + k] =
                        Math.abs(classes[k] - pClass) < 0.5 ? 1.0 : 0.0;
                }
            }
        }
        return fittedFinal.predict(new Matrix(metaFeatures));
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
        p.put("cv", cv);
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

    private Matrix extractRows(Matrix X, List<Integer> indices) {
        int n = indices.size();
        int m = X.cols();
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = X.get(indices.get(i), j);
            }
        }
        return new Matrix(data);
    }

    private Vector extractRows(Vector y, List<Integer> indices) {
        int n = indices.size();
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = y.get(indices.get(i));
        }
        return new Vector(data);
    }
}
