package org.sklearn.neural_network;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Multi-layer Perceptron regressor with backpropagation.
 */
public class MLPRegressor implements Predictor<Matrix, Vector, Vector> {

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
    private boolean earlyStopping;
    private double validationFraction;
    private int nIterNoChange;
    private long randomSeed;
    private boolean fitted;

    private List<Matrix> coefs;
    private List<Vector> intercepts;
    private int nFeatures;
    private int nLayers;
    private int nIter;
    private List<Double> lossCurve;
    private List<Double> validationScores;
    private int noImprovementCount;
    private double momentum;
    private boolean nesterovsMomentum;

    public MLPRegressor() {
        this(new int[]{100}, "relu", "adam", 0.0001, "constant", 0.001, 200, true, 1e-4, 200, true, 0.1, 10, 42);
    }

    public MLPRegressor(int[] hiddenLayerSizes, String activation, String solver,
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
    public MLPRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;
        this.nOutputs = 1;

        int[] layerUnits = new int[2 + hiddenLayerSizes.length];
        layerUnits[0] = m;
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            layerUnits[i + 1] = hiddenLayerSizes[i];
        }
        layerUnits[layerUnits.length - 1] = 1;
        this.nLayers = layerUnits.length;

        initWeights(layerUnits);
        Random rng = new Random(randomSeed);

        if ("sgd".equals(solver)) {
            fitSGD(X, y, layerUnits, rng);
        } else {
            fitAdam(X, y, layerUnits, rng);
        }

        this.fitted = true;
        return this;
    }

    private int nOutputs;

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

    private double[] forwardReg(Matrix X) {
        Matrix a = new Matrix(X);
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
        }
        double[] result = new double[a.rows()];
        for (int i = 0; i < a.rows(); i++) {
            result[i] = a.get(i, 0);
        }
        return result;
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

    private double mseLoss(double[] yTrue, double[] yPred) {
        double loss = 0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            loss += diff * diff;
        }
        return loss / yTrue.length;
    }

    private void fitSGD(Matrix X, Vector y, int[] layerUnits, Random rng) {
        int n = X.rows();
        lossCurve = new ArrayList<>();
        validationScores = new ArrayList<>();
        noImprovementCount = 0;
        int actualBatchSize = Math.min(batchSize, n);

        double[] yArr = new double[n];
        for (int i = 0; i < n; i++) {
            yArr[i] = y.get(i);
        }

        int nTrain = n;
        double[] yValArr = null;
        double[][] xValArr = null;
        if (earlyStopping && n > 10) {
            int nVal = Math.max(1, (int) (n * validationFraction));
            nTrain = n - nVal;
            xValArr = new double[nVal][nFeatures];
            yValArr = new double[nVal];
            for (int i = 0; i < nVal; i++) {
                yValArr[i] = yArr[i];
                for (int f = 0; f < nFeatures; f++) {
                    xValArr[i][f] = X.get(i, f);
                }
            }
        }

        int[] idx = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            idx[i] = earlyStopping ? i + (n - nTrain) : i;
        }
        double bestLoss = Double.MAX_VALUE;
        double lr = learningRateInit;

        for (int iter = 0; iter < maxIter; iter++) {
            if (shuffle) {
                shuffleArray(idx, rng);
            }

            double totalLoss = 0;
            for (int start = 0; start < nTrain; start += actualBatchSize) {
                int end = Math.min(start + actualBatchSize, nTrain);
                int bs = end - start;

                double[][] xBatch = new double[bs][nFeatures];
                double[] yBatch = new double[bs];
                for (int i = start; i < end; i++) {
                    int ri = idx[i];
                    yBatch[i - start] = yArr[ri];
                    for (int f = 0; f < nFeatures; f++) {
                        xBatch[i - start][f] = X.get(ri, f);
                    }
                }

                totalLoss += backpropAndUpdateReg(xBatch, yBatch, bs, lr);
            }

            double avgLoss = totalLoss / nTrain;
            lossCurve.add(avgLoss);

            if (earlyStopping && xValArr != null) {
                double valLoss = scoreValReg(xValArr, yValArr);
                validationScores.add(valLoss);
                if (valLoss < bestLoss - tol) {
                    bestLoss = valLoss;
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

    private void fitAdam(Matrix X, Vector y, int[] layerUnits, Random rng) {
        int n = X.rows();
        lossCurve = new ArrayList<>();
        validationScores = new ArrayList<>();
        noImprovementCount = 0;
        int actualBatchSize = Math.min(batchSize, n);

        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

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

        double[] yArr = new double[n];
        for (int i = 0; i < n; i++) {
            yArr[i] = y.get(i);
        }

        int nTrain = n;
        double[] yValArr = null;
        double[][] xValArr = null;
        if (earlyStopping && n > 10) {
            int nVal = Math.max(1, (int) (n * validationFraction));
            nTrain = n - nVal;
            xValArr = new double[nVal][nFeatures];
            yValArr = new double[nVal];
            for (int i = 0; i < nVal; i++) {
                yValArr[i] = yArr[i];
                for (int f = 0; f < nFeatures; f++) {
                    xValArr[i][f] = X.get(i, f);
                }
            }
        }

        int[] idx = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            idx[i] = earlyStopping ? i + (n - nTrain) : i;
        }
        double bestLoss = Double.MAX_VALUE;
        double lr = learningRateInit;
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

                double[][] xBatch = new double[bs][nFeatures];
                double[] yBatch = new double[bs];
                for (int i = start; i < end; i++) {
                    int ri = idx[i];
                    yBatch[i - start] = yArr[ri];
                    for (int f = 0; f < nFeatures; f++) {
                        xBatch[i - start][f] = X.get(ri, f);
                    }
                }

                totalLoss += backpropagateReg(xBatch, yBatch);

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
                            coefs.get(i).set(r, c, coefs.get(i).get(r, c) - lrT * mVal / (Math.sqrt(vVal) + eps));
                        }
                    }
                    for (int c = 0; c < gb.size(); c++) {
                        double g = gb.get(c);
                        double mVal = beta1 * mb.get(i).get(c) + (1 - beta1) * g;
                        double vVal = beta2 * vb.get(i).get(c) + (1 - beta2) * g * g;
                        mb.get(i).set(c, mVal);
                        vb.get(i).set(c, vVal);
                        intercepts.get(i).set(c, intercepts.get(i).get(c) - lrT * mVal / (Math.sqrt(vVal) + eps));
                    }
                }
            }

            double avgLoss = totalLoss / nTrain;
            lossCurve.add(avgLoss);

            if (earlyStopping && xValArr != null) {
                double valLoss = scoreValReg(xValArr, yValArr);
                validationScores.add(valLoss);
                if (valLoss < bestLoss - tol) {
                    bestLoss = valLoss;
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

    private double backpropAndUpdateReg(double[][] xBatch, double[] yBatch, int bs, double lr) {
        double loss = backpropagateReg(xBatch, yBatch);
        for (int i = 0; i < coefs.size(); i++) {
            Matrix gw = coefGrads.get(i);
            Vector gb = interGrads.get(i);
            for (int r = 0; r < gw.rows(); r++) {
                for (int c = 0; c < gw.cols(); c++) {
                    coefs.get(i).set(r, c, coefs.get(i).get(r, c) - lr * (gw.get(r, c) + alpha * coefs.get(i).get(r, c)));
                }
            }
            for (int c = 0; c < gb.size(); c++) {
                intercepts.get(i).set(c, intercepts.get(i).get(c) - lr * gb.get(c));
            }
        }
        return loss * bs;
    }

    private double backpropagateReg(double[][] xBatch, double[] yBatch) {
        int bs = xBatch.length;
        Matrix xMat = new Matrix(xBatch);

        // Forward pass with stored activations
        List<Matrix> activations = new ArrayList<>();
        activations.add(new Matrix(xMat));
        Matrix a = new Matrix(xMat);
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

        double[] preds = new double[bs];
        for (int i = 0; i < bs; i++) {
            preds[i] = a.get(i, 0);
        }
        double loss = mseLoss(yBatch, preds);

        // delta = 2*(y_pred - y_true) / bs  (gradient of MSE)
        double[] deltaArr = new double[bs];
        for (int i = 0; i < bs; i++) {
            deltaArr[i] = 2.0 * (a.get(i, 0) - yBatch[i]) / bs;
        }
        Matrix delta = new Matrix(new double[][]{deltaArr}).transpose();

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
                gb[j] = sum;
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

    private double scoreValReg(double[][] XVal, double[] yVal) {
        double[] preds = predictRaw(new Matrix(XVal));
        return mseLoss(yVal, preds);
    }

    private double[] predictRaw(Matrix X) {
        double[] result = forwardReg(X);
        return result;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "MLPRegressor");
        double[] preds = forwardReg(X);
        return new Vector(preds);
    }

    @Override
    public double score(Matrix X, Vector y) {
        Vector pred = predict(X);
        double ssRes = 0, ssTot = 0, yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double d = y.get(i) - pred.get(i); ssRes += d * d;
            double dm = y.get(i) - yMean; ssTot += dm * dm;
        }
        if (ssTot == 0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public int getNIter() {
        return nIter;
    }

    public List<Double> getLossCurve() {
        return lossCurve;
    }

    public boolean isFitted() {
        return fitted;
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
        p.put("max_iter", maxIter);
        return Collections.unmodifiableMap(p);
    }
}
