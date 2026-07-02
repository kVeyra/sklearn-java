package org.sklearn.neural_network;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Multi-layer Perceptron classifier with backpropagation.
 */
public class MLPClassifier implements Predictor<Matrix, Vector, Vector> {

    private int[] hiddenLayerSizes;
    private String activation;
    private String solver;
    private double alpha;
    private String learningRate;
    private double learningRateInit;
    private int maxIter;
    private boolean shuffle;
    private double tol;
    private int batchSize;
    private double momentum;
    private boolean nesterovsMomentum;
    private boolean earlyStopping;
    private double validationFraction;
    private int nIterNoChange;
    private long randomSeed;
    private boolean fitted;

    private List<Matrix> coefs;
    private List<Vector> intercepts;
    private int[] classes;
    private int nFeatures;
    private int nLayers;
    private int nOutputs;
    private int nIter;
    private List<Double> lossCurve;
    private List<Double> validationScores;
    private int noImprovementCount;

    public MLPClassifier() {
        this(new int[]{100}, "relu", "adam", 0.0001, "constant", 0.001, 200, true, 1e-4, 200, true, 0.1, 10, 42);
    }

    public MLPClassifier(int[] hiddenLayerSizes, String activation, String solver,
                         double alpha, String learningRate, double learningRateInit,
                         int maxIter, boolean shuffle, double tol, int batchSize,
                         boolean earlyStopping, double validationFraction,
                         int nIterNoChange, long randomSeed) {
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.activation = activation;
        this.solver = solver;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.learningRateInit = learningRateInit;
        this.maxIter = maxIter;
        this.shuffle = shuffle;
        this.tol = tol;
        this.batchSize = batchSize;
        this.momentum = 0.9;
        this.nesterovsMomentum = true;
        this.earlyStopping = earlyStopping;
        this.validationFraction = validationFraction;
        this.nIterNoChange = nIterNoChange;
        this.randomSeed = randomSeed;
    }

    @Override
    public MLPClassifier fit(Matrix X, Vector y) {
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
        this.nOutputs = classes.length;

        int[] layerUnits = new int[2 + hiddenLayerSizes.length];
        layerUnits[0] = m;
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            layerUnits[i + 1] = hiddenLayerSizes[i];
        }
        layerUnits[layerUnits.length - 1] = nOutputs;
        this.nLayers = layerUnits.length;

        // one-hot encode y
        Matrix yOneHot = new Matrix(n, nOutputs);
        for (int i = 0; i < n; i++) {
            int c = indexOf(classes, (int) y.get(i));
            yOneHot.set(i, c, 1.0);
        }

        initWeights(layerUnits);
        Random rng = new Random(randomSeed);

        if ("sgd".equals(solver)) {
            fitSGD(X, yOneHot, layerUnits, rng);
        } else {
            fitAdam(X, yOneHot, layerUnits, rng);
        }

        this.fitted = true;
        return this;
    }

    private void initWeights(int[] layerUnits) {
        coefs = new ArrayList<>();
        intercepts = new ArrayList<>();
        Random rng = new Random(randomSeed + 42);

        for (int i = 0; i < layerUnits.length - 1; i++) {
            int fanIn = layerUnits[i];
            int fanOut = layerUnits[i + 1];
            double factor = "logistic".equals(activation) ? 2.0 : 6.0;
            double bound = Math.sqrt(factor / (fanIn + fanOut));

            double[][] w = new double[fanIn][fanOut];
            double[] b = new double[fanOut];
            for (int r = 0; r < fanIn; r++) {
                for (int c = 0; c < fanOut; c++) {
                    w[r][c] = rng.nextDouble() * 2 * bound - bound;
                }
            }
            for (int c = 0; c < fanOut; c++) {
                b[c] = rng.nextDouble() * 2 * bound - bound;
            }
            coefs.add(new Matrix(w));
            intercepts.add(new Vector(b));
        }
    }

    private Matrix forward(Matrix X, List<Matrix> activations) {
        activations.clear();
        activations.add(new Matrix(X));
        Matrix a = X;
        for (int i = 0; i < coefs.size(); i++) {
            a = a.multiply(coefs.get(i));
            for (int r = 0; r < a.rows(); r++) {
                for (int c = 0; c < a.cols(); c++) {
                    a.set(r, c, a.get(r, c) + intercepts.get(i).get(c));
                }
            }
            if (i < coefs.size() - 1) {
                a = applyActivation(a);
            }
            activations.add(new Matrix(a));
        }
        // softmax for last layer
        a = softmax(a);
        activations.set(activations.size() - 1, new Matrix(a));
        return a;
    }

    private Matrix applyActivation(Matrix a) {
        Matrix r = new Matrix(a.rows(), a.cols());
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double v = a.get(i, j);
                switch (activation) {
                    case "relu": r.set(i, j, v > 0 ? v : 0); break;
                    case "tanh": r.set(i, j, Math.tanh(v)); break;
                    case "logistic": r.set(i, j, 1.0 / (1.0 + Math.exp(-v))); break;
                    default: r.set(i, j, v); break;
                }
            }
        }
        return r;
    }

    private Matrix activationDerivative(Matrix a) {
        Matrix r = new Matrix(a.rows(), a.cols());
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double v = a.get(i, j);
                switch (activation) {
                    case "relu": r.set(i, j, v > 0 ? 1 : 0); break;
                    case "tanh": r.set(i, j, 1 - v * v); break;
                    case "logistic": r.set(i, j, v * (1 - v)); break;
                    default: r.set(i, j, 1); break;
                }
            }
        }
        return r;
    }

    private Matrix softmax(Matrix a) {
        Matrix r = new Matrix(a.rows(), a.cols());
        for (int i = 0; i < a.rows(); i++) {
            double max = a.get(i, 0);
            for (int j = 1; j < a.cols(); j++) {
                if (a.get(i, j) > max) {
                    max = a.get(i, j);
                }
            }
            double sum = 0;
            for (int j = 0; j < a.cols(); j++) {
                double exp = Math.exp(a.get(i, j) - max);
                r.set(i, j, exp);
                sum += exp;
            }
            for (int j = 0; j < a.cols(); j++) {
                r.set(i, j, r.get(i, j) / sum);
            }
        }
        return r;
    }

    private double crossEntropy(Matrix yTrue, Matrix yPred) {
        double loss = 0;
        int n = yTrue.rows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < yTrue.cols(); j++) {
                double p = Math.max(yPred.get(i, j), 1e-15);
                loss -= yTrue.get(i, j) * Math.log(p);
            }
        }
        return loss / n;
    }

    private void fitSGD(Matrix X, Matrix yOneHot, int[] layerUnits, Random rng) {
        int n = X.rows();
        lossCurve = new ArrayList<>();
        validationScores = new ArrayList<>();
        noImprovementCount = 0;
        int actualBatchSize = Math.min(batchSize, n);

        Matrix xVal = null, yVal = null;
        int nTrain = n;
        if (earlyStopping && n > 10) {
            int nVal = Math.max(1, (int) (n * validationFraction));
            nTrain = n - nVal;
            xVal = new Matrix(nVal, nFeatures);
            yVal = new Matrix(nVal, nOutputs);
            for (int i = 0; i < nVal; i++) {
                for (int f = 0; f < nFeatures; f++) {
                    xVal.set(i, f, X.get(i, f));
                }
                for (int c = 0; c < nOutputs; c++) {
                    yVal.set(i, c, yOneHot.get(i, c));
                }
            }
        }

        int[] idx = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            idx[i] = earlyStopping ? i + (n - nTrain) : i;
        }
        double bestLoss = Double.MAX_VALUE;

        for (int iter = 0; iter < maxIter; iter++) {
            if (shuffle) {
                shuffleArray(idx, rng);
            }

            double totalLoss = 0;
            for (int start = 0; start < nTrain; start += actualBatchSize) {
                int end = Math.min(start + actualBatchSize, nTrain);
                int bs = end - start;

                Matrix xBatch = new Matrix(bs, nFeatures);
                Matrix yBatch = new Matrix(bs, nOutputs);
                for (int i = start; i < end; i++) {
                    int ri = idx[i];
                    for (int f = 0; f < nFeatures; f++) {
                        xBatch.set(i - start, f, X.get(ri, f));
                    }
                    for (int c = 0; c < nOutputs; c++) {
                        yBatch.set(i - start, c, yOneHot.get(ri, c));
                    }
                }

                totalLoss += backpropAndUpdate(xBatch, yBatch, bs);
            }

            double avgLoss = totalLoss / nTrain;
            lossCurve.add(avgLoss);

            if (earlyStopping && xVal != null) {
                double valScore = scoreVal(xVal, yVal);
                validationScores.add(valScore);
                if (valScore < bestLoss - tol) {
                    bestLoss = valScore;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                if (avgLoss < bestLoss - tol) {
                    bestLoss = avgLoss;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            }

            if (noImprovementCount > nIterNoChange) {
                this.nIter = iter + 1;
                return;
            }
        }
        this.nIter = maxIter;
    }

    private void fitAdam(Matrix X, Matrix yOneHot, int[] layerUnits, Random rng) {
        int n = X.rows();
        lossCurve = new ArrayList<>();
        validationScores = new ArrayList<>();
        noImprovementCount = 0;
        int actualBatchSize = Math.min(batchSize, n);

        // Adam state
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double lr = learningRateInit;

        List<Matrix> mW = new ArrayList<>();
        List<Vector> mb = new ArrayList<>();
        List<Matrix> vW = new ArrayList<>();
        List<Vector> vb = new ArrayList<>();
        for (int i = 0; i < coefs.size(); i++) {
            mW.add(new Matrix(coefs.get(i).rows(), coefs.get(i).cols()));
            mb.add(new Vector(intercepts.get(i).size()));
            vW.add(new Matrix(coefs.get(i).rows(), coefs.get(i).cols()));
            vb.add(new Vector(intercepts.get(i).size()));
        }

        Matrix xVal = null;
        Matrix yVal = null;
        int nTrain = n;
        if (earlyStopping && n > 10) {
            int nVal = Math.max(1, (int) (n * validationFraction));
            nTrain = n - nVal;
            xVal = new Matrix(nVal, nFeatures);
            yVal = new Matrix(nVal, nOutputs);
            for (int i = 0; i < nVal; i++) {
                for (int f = 0; f < nFeatures; f++) {
                    xVal.set(i, f, X.get(i, f));
                }
                for (int c = 0; c < nOutputs; c++) {
                    yVal.set(i, c, yOneHot.get(i, c));
                }
            }
        }

        int[] idx = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            idx[i] = earlyStopping ? i + (n - nTrain) : i;
        }
        double bestLoss = Double.MAX_VALUE;
        int t = 0;

        for (int iter = 0; iter < maxIter; iter++) {
            if (shuffle) {
                shuffleArray(idx, rng);
            }

            double totalLoss = 0;
            for (int start = 0; start < nTrain; start += actualBatchSize) {
                int end = Math.min(start + actualBatchSize, nTrain);
                int bs = end - start;
                t++;

                Matrix xBatch = new Matrix(bs, nFeatures);
                Matrix yBatch = new Matrix(bs, nOutputs);
                for (int i = start; i < end; i++) {
                    int ri = idx[i];
                    for (int f = 0; f < nFeatures; f++) {
                        xBatch.set(i - start, f, X.get(ri, f));
                    }
                    for (int c = 0; c < nOutputs; c++) {
                        yBatch.set(i - start, c, yOneHot.get(ri, c));
                    }
                }

                totalLoss += backpropagate(xBatch, yBatch);

                // Adam update
                double lrT = lr * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));
                for (int i = 0; i < coefs.size(); i++) {
                    Matrix gw = coefGrads.get(i);
                    Vector gb = interGrads.get(i);
                    for (int r = 0; r < gw.rows(); r++) {
                        for (int c = 0; c < gw.cols(); c++) {
                            double g = gw.get(r, c);
                            double mVal = beta1 * mW.get(i).get(r, c) + (1 - beta1) * g;
                            double vVal = beta2 * vW.get(i).get(r, c) + (1 - beta2) * g * g;
                            mW.get(i).set(r, c, mVal);
                            vW.get(i).set(r, c, vVal);
                            double update = lrT * mVal / (Math.sqrt(vVal) + eps);
                            coefs.get(i).set(r, c, coefs.get(i).get(r, c) - update);
                        }
                    }
                    for (int c = 0; c < gb.size(); c++) {
                        double g = gb.get(c);
                        double mVal = beta1 * mb.get(i).get(c) + (1 - beta1) * g;
                        double vVal = beta2 * vb.get(i).get(c) + (1 - beta2) * g * g;
                        mb.get(i).set(c, mVal);
                        vb.get(i).set(c, vVal);
                        double update = lrT * mVal / (Math.sqrt(vVal) + eps);
                        intercepts.get(i).set(c, intercepts.get(i).get(c) - update);
                    }
                }
            }

            double avgLoss = totalLoss / nTrain;
            lossCurve.add(avgLoss);

            if (earlyStopping && xVal != null) {
                double valScore = scoreVal(xVal, yVal);
                validationScores.add(valScore);
                if (valScore < bestLoss - tol) {
                    bestLoss = valScore;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                if (avgLoss < bestLoss - tol) {
                    bestLoss = avgLoss;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            }

            if (noImprovementCount > nIterNoChange) {
                this.nIter = iter + 1;
                return;
            }
        }
        this.nIter = maxIter;
    }

    private List<Matrix> coefGrads;
    private List<Vector> interGrads;

    private double backpropAndUpdate(Matrix xBatch, Matrix yBatch, int bs) {
        double loss = backpropagate(xBatch, yBatch);

        double lr = "constant".equals(learningRate) ? learningRateInit : learningRateInit;
        for (int i = 0; i < coefs.size(); i++) {
            Matrix gw = coefGrads.get(i);
            Vector gb = interGrads.get(i);
            for (int r = 0; r < gw.rows(); r++) {
                for (int c = 0; c < gw.cols(); c++) {
                    double update = lr * (gw.get(r, c) + alpha * coefs.get(i).get(r, c));
                    coefs.get(i).set(r, c, coefs.get(i).get(r, c) - update);
                }
            }
            for (int c = 0; c < gb.size(); c++) {
                intercepts.get(i).set(c, intercepts.get(i).get(c) - lr * gb.get(c));
            }
        }
        return loss * bs;
    }

    private double backpropagate(Matrix xBatch, Matrix yBatch) {
        List<Matrix> activations = new ArrayList<>();
        Matrix output = forward(xBatch, activations);

        int bs = xBatch.rows();

        // Compute loss
        double loss = crossEntropy(yBatch, output);

        // Backprop: delta = output - y for softmax + cross-entropy
        Matrix delta = output.subtract(yBatch);

        // Compute gradients
        coefGrads = new ArrayList<>();
        interGrads = new ArrayList<>();

        for (int layer = coefs.size() - 1; layer >= 0; layer--) {
            Matrix aPrev = activations.get(layer);
            Matrix gw = aPrev.transpose().multiply(delta);
            double[] gb = new double[delta.cols()];
            for (int j = 0; j < delta.cols(); j++) {
                double sum = 0;
                for (int i = 0; i < delta.rows(); i++) {
                    sum += delta.get(i, j);
                }
                gb[j] = sum / bs;
            }
            for (int r = 0; r < gw.rows(); r++) {
                for (int c = 0; c < gw.cols(); c++) {
                    gw.set(r, c, gw.get(r, c) / bs);
                }
            }
            coefGrads.add(0, gw);
            interGrads.add(0, new Vector(gb));

            if (layer > 0) {
                Matrix w = coefs.get(layer);
                Matrix deltaPrev = delta.multiply(w.transpose());
                Matrix act = activations.get(layer);
                Matrix deriv = activationDerivative(act);
                for (int i = 0; i < deltaPrev.rows(); i++) {
                    for (int j = 0; j < deltaPrev.cols(); j++) {
                        deltaPrev.set(i, j, deltaPrev.get(i, j) * deriv.get(i, j));
                    }
                }
                delta = deltaPrev;
            }
        }
        return loss;
    }

    private double scoreVal(Matrix X, Matrix yOneHot) {
        List<Matrix> activations = new ArrayList<>();
        Matrix output = forward(X, activations);
        double loss = crossEntropy(yOneHot, output);
        return loss;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "MLPClassifier");
        List<Matrix> activations = new ArrayList<>();
        Matrix output = forward(X, activations);
        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int j = 1; j < output.cols(); j++) {
                if (output.get(i, j) > output.get(i, best)) {
                    best = j;
                }
            }
            preds[i] = classes[best];
        }
        return new Vector(preds);
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

    public double[] predictProbas(Matrix X) {
        Validation.checkFitted(fitted, "MLPClassifier");
        List<Matrix> activations = new ArrayList<>();
        Matrix output = forward(X, activations);
        double[] probas = new double[X.rows() * output.cols()];
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {
                probas[i * output.cols() + j] = output.get(i, j);
            }
        }
        return probas;
    }

    public int getNIter() {
        return nIter;
    }

    public List<Double> getLossCurve() {
        return lossCurve;
    }

    public int[] getClasses() {
        return classes;
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

    private void shuffleArray(int[] arr, Random rng) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("hidden_layer_sizes", hiddenLayerSizes);
        p.put("activation", activation);
        p.put("solver", solver);
        p.put("alpha", alpha);
        p.put("learning_rate", learningRate);
        p.put("learning_rate_init", learningRateInit);
        p.put("max_iter", maxIter);
        p.put("batch_size", batchSize);
        return Collections.unmodifiableMap(p);
    }
}
