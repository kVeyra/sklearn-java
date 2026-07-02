package org.sklearn.linear_model;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * SGD classifier.
 */
public class SGDClassifier implements Predictor<Matrix, Vector, Vector> {

    private String loss;
    private String penalty;
    private double alpha;
    private double l1Ratio;
    private boolean fitIntercept;
    private int maxIter;
    private double tol;
    private String learningRate;
    private double eta0;
    private double powerT;
    private boolean earlyStopping;
    private double validationFraction;
    private int nIterNoChange;
    private boolean shuffle;
    private long randomState;
    private boolean average;
    private boolean warmStart;

    private Vector[] coefs;
    private Vector intercepts;
    private int[] classes;
    private int nFeatures;
    private boolean fitted;
    private int nIter;

    public SGDClassifier() {
        this("hinge", "l2", 0.0001, 0.15, true, 1000, 1e-3,
            "optimal", 0.0, 0.5, false, 0.1, 5, true, 42, false, false);
    }

    @SuppressWarnings("checkstyle:parameternumber")
    public SGDClassifier(String loss, String penalty, double alpha, double l1Ratio,
                          boolean fitIntercept, int maxIter, double tol,
                          String learningRate, double eta0, double powerT,
                          boolean earlyStopping, double validationFraction,
                          int nIterNoChange, boolean shuffle, long randomState,
                          boolean average, boolean warmStart) {
        this.loss = loss;
        this.penalty = penalty;
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.learningRate = learningRate;
        this.eta0 = eta0;
        this.powerT = powerT;
        this.earlyStopping = earlyStopping;
        this.validationFraction = validationFraction;
        this.nIterNoChange = nIterNoChange;
        this.shuffle = shuffle;
        this.randomState = randomState;
        this.average = average;
        this.warmStart = warmStart;
    }

    @Override
    public SGDClassifier fit(Matrix X, Vector y) {
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
        if (nClasses < 2) {
            throw new IllegalArgumentException("SGDClassifier requires at least 2 classes");
        }

        if (nClasses == 2) {
            fitBinary(X, y, classes[0], classes[1]);
        } else {
            fitMulticlass(X, y, nClasses);
        }

        this.fitted = true;
        return this;
    }

    private void fitBinary(Matrix X, Vector y, int negLabel, int posLabel) {
        int n = X.rows();
        int m = X.cols();

        // Encode to {-1, +1}
        double[] yEncoded = new double[n];
        for (int i = 0; i < n; i++) {
            yEncoded[i] = (y.get(i) == posLabel) ? 1.0 : -1.0;
        }

        double[] w = new double[m];
        double bias = 0.0;
        if (warmStart && coefs != null && coefs.length > 0) {
            for (int j = 0; j < m; j++) {
                w[j] = coefs[0].get(j);
            }
            bias = intercepts.get(0);
        }

        double[] avgW = new double[m];
        double avgBias = 0.0;
        int t = 1;

        for (int epoch = 0; epoch < maxIter; epoch++) {
            List<Integer> idxList = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                idxList.add(i);
            }
            if (shuffle) {
                Collections.shuffle(idxList, new Random(randomState + epoch));
            }

            for (int si = 0; si < n; si++) {
                int i = idxList.get(si);
                double xiNormSq = 0.0;
                for (int j = 0; j < m; j++) {
                    double v = X.get(i, j);
                    xiNormSq += v * v;
                }

                double margin = bias;
                for (int j = 0; j < m; j++) {
                    margin += X.get(i, j) * w[j];
                }
                margin *= yEncoded[i];

                double dloss = lossGradient(yEncoded[i], margin);
                if (dloss == 0.0) {
                    t++;
                    continue;
                }

                double eta = learningRate(t);
                double weightDecay = 1.0;
                if (penalty.equals("l2")) {
                    weightDecay = 1.0 - eta * alpha;
                } else if (penalty.equals("elasticnet")) {
                    weightDecay = 1.0 - eta * alpha * (1.0 - l1Ratio);
                }

                boolean isL1 = penalty.equals("l1") || penalty.equals("elasticnet");
                double l1Mult = penalty.equals("l1") ? 1.0 : l1Ratio;

                for (int j = 0; j < m; j++) {
                    w[j] *= weightDecay;
                    w[j] -= eta * dloss * X.get(i, j);

                    if (isL1 && eta * alpha * l1Mult > Math.abs(w[j])) {
                        w[j] = 0.0;
                    } else if (isL1 && w[j] > 0) {
                        w[j] -= eta * alpha * l1Mult;
                    } else if (isL1) {
                        w[j] += eta * alpha * l1Mult;
                    }
                }

                bias -= eta * dloss;

                if (average) {
                    double mu = 1.0 / t;
                    for (int j = 0; j < m; j++) {
                        avgW[j] += mu * (w[j] - avgW[j]);
                    }
                    avgBias += mu * (bias - avgBias);
                }

                t++;
            }

            if (tol > 0 && epoch > 0) {
                double change = 0.0;
                for (int j = 0; j < m; j++) {
                    change += Math.abs(w[j]);
                }
                if (change / m < tol) {
                    nIter = epoch + 1;
                    break;
                }
            }
        }

        if (t > n * maxIter) {
            nIter = maxIter;
        } else if (nIter == 0) {
            nIter = maxIter;
        }

        if (average) {
            coefs = new Vector[]{new Vector(avgW)};
            intercepts = new Vector(new double[]{avgBias});
        } else {
            coefs = new Vector[]{new Vector(w)};
            intercepts = new Vector(new double[]{bias});
        }
    }

    private void fitMulticlass(Matrix X, Vector y, int nClasses) {
        int n = X.rows();
        int m = X.cols();

        coefs = new Vector[nClasses];
        intercepts = new Vector(new double[nClasses]);

        for (int c = 0; c < nClasses; c++) {
            int posLabel = classes[c];
            double[] yBin = new double[n];
            for (int i = 0; i < n; i++) {
                yBin[i] = (y.get(i) == posLabel) ? 1.0 : -1.0;
            }

            double[] w = new double[m];
            double bias = 0.0;
            int t = 1;

            for (int epoch = 0; epoch < maxIter; epoch++) {
                List<Integer> idxList = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    idxList.add(i);
                }
                if (shuffle) {
                    Collections.shuffle(idxList, new Random(randomState + epoch + c * 1000));
                }

                for (int si = 0; si < n; si++) {
                    int i = idxList.get(si);
                    double margin = bias;
                    for (int j = 0; j < m; j++) {
                        margin += X.get(i, j) * w[j];
                    }
                    margin *= yBin[i];

                    double dloss = lossGradient(yBin[i], margin);
                    if (dloss == 0.0) {
                        t++;
                        continue;
                    }

                    double eta = learningRate(t);

                    for (int j = 0; j < m; j++) {
                        if (penalty.equals("l2")) {
                            w[j] *= (1.0 - eta * alpha);
                        } else if (penalty.equals("elasticnet")) {
                            w[j] *= (1.0 - eta * alpha * (1.0 - l1Ratio));
                        }
                        w[j] -= eta * dloss * X.get(i, j);
                    }
                    bias -= eta * dloss;
                    t++;
                }
            }

            coefs[c] = new Vector(w);
            intercepts.set(c, bias);
        }
    }

    private double lossGradient(double y, double margin) {
        switch (loss) {
            case "hinge":
                return margin < 1.0 ? -y : 0.0;
            case "squared_hinge":
                if (margin < 1.0) {
                    return -2.0 * y * (1.0 - margin);
                }
                return 0.0;
            case "perceptron":
                return margin < 0.0 ? -y : 0.0;
            case "log_loss":
                double p = sigmoid(margin);
                return p - 1.0;
            case "modified_huber":
                if (margin < -1.0) {
                    return -4.0 * y;
                }
                if (margin < 1.0) {
                    return -2.0 * y * (1.0 - margin);
                }
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown loss: " + loss);
        }
    }

    private double learningRate(int t) {
        switch (learningRate) {
            case "constant":
                return eta0 > 0 ? eta0 : 0.01;
            case "invscaling":
                return eta0 / Math.pow(t, powerT);
            case "optimal":
                if (alpha <= 0) {
                    return Math.pow(t, -0.5);
                }
                return 1.0 / (alpha * (1.0 + t - 1.0));
            case "pa1":
            case "pa2":
                return eta0;
            default:
                return 0.01;
        }
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "SGDClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException("Feature dimension mismatch: expected " + nFeatures + " got " + X.cols());
        }

        int n = X.rows();
        int nClasses = classes.length;
        double[] preds = new double[n];

        for (int i = 0; i < n; i++) {
            if (nClasses == 2) {
                double score = intercepts.get(0);
                for (int j = 0; j < nFeatures; j++) {
                    score += X.get(i, j) * coefs[0].get(j);
                }
                preds[i] = score > 0 ? classes[1] : classes[0];
            } else {
                double bestScore = Double.NEGATIVE_INFINITY;
                int bestIdx = 0;
                for (int c = 0; c < nClasses; c++) {
                    double score = intercepts.get(c);
                    for (int j = 0; j < nFeatures; j++) {
                        score += X.get(i, j) * coefs[c].get(j);
                    }
                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx = c;
                    }
                }
                preds[i] = classes[bestIdx];
            }
        }
        return new Vector(preds);
    }

    public Vector predictProba(Matrix X) {
        Validation.checkFitted(fitted, "SGDClassifier");
        if (!loss.equals("log_loss") && !loss.equals("modified_huber")) {
            throw new UnsupportedOperationException(
                "predictProba is only supported for loss='log_loss' or 'modified_huber', got: " + loss);
        }

        int n = X.rows();
        double[] probs = new double[n];
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            double s = intercepts.get(0);
            for (int j = 0; j < nFeatures; j++) {
                s += X.get(i, j) * coefs[0].get(j);
            }
            scores[i] = s;
        }

        if (loss.equals("log_loss")) {
            for (int i = 0; i < n; i++) {
                probs[i] = 1.0 / (1.0 + Math.exp(-scores[i]));
            }
        } else {
            for (int i = 0; i < n; i++) {
                double s = Math.max(-1.0, Math.min(1.0, scores[i]));
                probs[i] = (s + 1.0) / 2.0;
            }
        }
        return new Vector(probs);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "SGDClassifier");
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

    public Vector[] getCoefs() {
        return coefs;
    }
    public Vector getIntercepts() {
        return intercepts;
    }
    public int[] getClasses() {
        return classes;
    }
    public int getNIter() {
        return nIter;
    }
    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss); p.put("penalty", penalty); p.put("alpha", alpha);
        p.put("l1_ratio", l1Ratio); p.put("fit_intercept", fitIntercept);
        p.put("max_iter", maxIter); p.put("tol", tol);
        p.put("learning_rate", learningRate); p.put("eta0", eta0);
        p.put("power_t", powerT);
        return Collections.unmodifiableMap(p);
    }

    private static double sigmoid(double z) {
        if (z > 40) {
            return 1.0;
        }
        if (z < -40) {
            return 0.0;
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }
}
