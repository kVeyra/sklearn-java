package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * AdaBoost classifier.
 *
 * <p>Fits a sequence of weak learners (default: decision stumps with
 * {@code maxDepth=1}) on weighted versions of the training data. At each
 * iteration sample weights are updated to give more weight to misclassified
 * samples. Prediction uses weighted majority voting (SAMME algorithm).
 *
 * <p>Mirrors {@code sklearn.ensemble.AdaBoostClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * AdaBoostClassifier ada = new AdaBoostClassifier(50, 1, 42);
 * ada.fit(X, y);
 * Vector preds = ada.predict(X_test);
 * }</pre>
 */
public class AdaBoostClassifier implements Predictor<Matrix, Vector, Vector> {

    private int nEstimators;
    private int maxDepth;
    private double learningRate;
    private long seed;
    private boolean fitted;
    private List<DecisionTreeClassifier> estimators;
    private double[] estimatorWeights;
    private int[] classes;
    private int nFeatures;

    /**
     * Create an AdaBoost classifier.
     *
     * @param nEstimators number of weak learners
     * @param maxDepth    max depth of each weak learner (default 1 = stump)
     * @param seed        random seed
     */
    public AdaBoostClassifier(int nEstimators, int maxDepth, long seed) {
        this(nEstimators, maxDepth, 1.0, seed);
    }

    /**
     * Create an AdaBoost classifier with full control.
     *
     * @param nEstimators  number of weak learners
     * @param maxDepth     max depth of each weak learner
     * @param learningRate learning rate (shrinks contribution of each learner)
     * @param seed         random seed
     */
    public AdaBoostClassifier(int nEstimators, int maxDepth,
                               double learningRate, long seed) {
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be >= 1");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.learningRate = learningRate;
        this.seed = seed;
    }

    @Override
    public AdaBoostClassifier fit(Matrix X, Vector y) {
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

        double[] sampleWeights = new double[n];
        Arrays.fill(sampleWeights, 1.0 / n);

        estimators = new ArrayList<>();
        estimatorWeights = new double[nEstimators];
        double epsilon = 1e-10;

        for (int t = 0; t < nEstimators; t++) {
            for (int i = 0; i < n; i++) {
                if (sampleWeights[i] == 0) {
                    sampleWeights[i] = epsilon;
                }
            }

            Object[] bootData = bootstrapSample(X, y, sampleWeights, t);
            DecisionTreeClassifier stump = new DecisionTreeClassifier(
                maxDepth, 2, 1, "gini");
            stump.fit((Matrix) bootData[0], (Vector) bootData[1]);

            Vector preds = stump.predict(X);
            double error = 0.0;
            for (int i = 0; i < n; i++) {
                if (Math.abs(preds.get(i) - y.get(i)) >= 0.5) {
                    error += sampleWeights[i];
                }
            }
            error /= sum(sampleWeights);

            if (error > 1.0 - 1.0 / nClasses) {
                estimatorWeights[t] = 0.0;
                estimators.add(stump);
                continue;
            }

            double alpha;
            if (nClasses == 2) {
                alpha = learningRate * 0.5 * Math.log((1 - error) / Math.max(error, epsilon));
            } else {
                alpha = learningRate * (Math.log((1 - error) / Math.max(error, epsilon))
                    + Math.log(nClasses - 1));
            }

            estimatorWeights[t] = alpha;
            estimators.add(stump);

            double sumW = 0.0;
            for (int i = 0; i < n; i++) {
                boolean misclassified = Math.abs(preds.get(i) - y.get(i)) >= 0.5;
                sampleWeights[i] *= Math.exp(alpha * (misclassified ? 1 : -1));
                sampleWeights[i] = Math.max(sampleWeights[i], epsilon);
                sumW += sampleWeights[i];
            }
            for (int i = 0; i < n; i++) {
                sampleWeights[i] /= sumW;
            }

            if (error == 0) {
                break;
            }
        }

        fitted = true;
        return this;
    }

    private Object[] bootstrapSample(Matrix X, Vector y, double[] weights, long iterSeed) {
        int n = X.rows();
        int m = X.cols();
        RandomGenerator rng = new RandomGenerator(seed + iterSeed * 1000);

        double[] cumSum = new double[n];
        cumSum[0] = weights[0];
        for (int i = 1; i < n; i++) {
            cumSum[i] = cumSum[i - 1] + weights[i];
        }
        double totalW = cumSum[n - 1];

        Matrix bootX = new Matrix(n, m);
        Vector bootY = new Vector(n);
        for (int i = 0; i < n; i++) {
            double r = rng.nextDouble() * totalW;
            int idx = lowerBound(cumSum, r);
            idx = Math.min(idx, n - 1);
            for (int j = 0; j < m; j++) {
                bootX.set(i, j, X.get(idx, j));
            }
            bootY.set(i, y.get(idx));
        }
        return new Object[]{bootX, bootY};
    }

    private int lowerBound(double[] arr, double val) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (arr[mid] < val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private double sum(double[] arr) {
        double s = 0;
        for (double v : arr) {
            s += v;
        }
        return s;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "AdaBoostClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        int nClasses = classes.length;

        if (nClasses == 2) {
            double[] weightedScores = new double[n];
            for (int t = 0; t < estimators.size(); t++) {
                if (estimatorWeights[t] == 0) {
                    continue;
                }
                Vector preds = estimators.get(t).predict(X);
                for (int i = 0; i < n; i++) {
                    weightedScores[i] += estimatorWeights[t]
                        * (preds.get(i) == classes[1] ? 1 : -1);
                }
            }
            double[] out = new double[n];
            for (int i = 0; i < n; i++) {
                out[i] = weightedScores[i] >= 0 ? classes[1] : classes[0];
            }
            return new Vector(out);
        } else {
            double[][] votes = new double[n][nClasses];
            for (int t = 0; t < estimators.size(); t++) {
                if (estimatorWeights[t] == 0) {
                    continue;
                }
                Vector preds = estimators.get(t).predict(X);
                for (int i = 0; i < n; i++) {
                    int predClass = (int) preds.get(i);
                    int idx = classIndex(predClass);
                    votes[i][idx] += estimatorWeights[t];
                }
            }
            double[] out = new double[n];
            for (int i = 0; i < n; i++) {
                out[i] = classes[argmax(votes[i])];
            }
            return new Vector(out);
        }
    }

    /**
     * Predict class probabilities based on normalized weighted votes.
     */
    public Vector predictProba(Matrix X) {
        Validation.checkFitted(fitted, "AdaBoostClassifier");
        Validation.checkMatrix(X, -1);

        int n = X.rows();
        int nClasses = classes.length;

        double[] probs = new double[n];

        if (nClasses == 2) {
            double maxAbsWeight = 0;
            double[] scores = new double[n];
            for (int t = 0; t < estimators.size(); t++) {
                if (estimatorWeights[t] == 0) {
                    continue;
                }
                Vector preds = estimators.get(t).predict(X);
                double absW = Math.abs(estimatorWeights[t]);
                maxAbsWeight += absW;
                for (int i = 0; i < n; i++) {
                    scores[i] += estimatorWeights[t]
                        * (preds.get(i) == classes[1] ? 1 : -1);
                }
            }
            for (int i = 0; i < n; i++) {
                probs[i] = maxAbsWeight > 0
                    ? (scores[i] / maxAbsWeight + 1.0) / 2.0
                    : 0.5;
                probs[i] = Math.max(0.0, Math.min(1.0, probs[i]));
            }
        } else {
            double[][] votes = new double[n][nClasses];
            for (int t = 0; t < estimators.size(); t++) {
                if (estimatorWeights[t] == 0) {
                    continue;
                }
                Vector preds = estimators.get(t).predict(X);
                for (int i = 0; i < n; i++) {
                    int idx = classIndex((int) preds.get(i));
                    votes[i][idx] += estimatorWeights[t];
                }
            }
            for (int i = 0; i < n; i++) {
                double minVote = Double.MAX_VALUE, maxVote = -Double.MAX_VALUE;
                for (int k = 0; k < nClasses; k++) {
                    if (votes[i][k] < minVote) {
                        minVote = votes[i][k];
                    }
                    if (votes[i][k] > maxVote) {
                        maxVote = votes[i][k];
                    }
                }
                double range = maxVote - minVote;
                if (range > 0) {
                    probs[i] = (votes[i][classIndex(classes[1])] - minVote) / range;
                } else {
                    probs[i] = 0.5;
                }
            }
        }
        return new Vector(probs);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "AdaBoostClassifier");
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

    private int classIndex(int cls) {
        for (int i = 0; i < classes.length; i++) {
            if (classes[i] == cls) {
                return i;
            }
        }
        return 0;
    }

    private int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) {
                best = i;
            }
        }
        return best;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("n_estimators", nEstimators);
        params.put("max_depth", maxDepth);
        params.put("learning_rate", learningRate);
        return Collections.unmodifiableMap(params);
    }
}
