package org.sklearn.ensemble;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Gradient Boosting for classification.
 *
 * <p>Builds an additive ensemble of regression trees in a forward stage-wise
 * manner, optimizing binomial/multinomial deviance.
 *
 * <p>Mirrors {@code sklearn.ensemble.GradientBoostingClassifier}.
 */
public class GradientBoostingClassifier implements Predictor<Matrix, Vector, Vector> {

    private String loss;
    private double learningRate;
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private double subsample;
    private long seed;

    private List<List<DecisionTreeRegressor>> estimators;
    private double[] initPredictions;
    private int nClasses;
    private int[] classes;

    public GradientBoostingClassifier() {
        this("deviance", 0.1, 100, 3, 2, 1, 1.0, 42);
    }

    public GradientBoostingClassifier(int nEstimators, int maxDepth, long seed) {
        this("deviance", 0.1, nEstimators, maxDepth, 2, 1, 1.0, seed);
    }

    public GradientBoostingClassifier(String loss, double learningRate, int nEstimators,
                                       int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                                       double subsample, long seed) {
        this.loss = loss;
        this.learningRate = learningRate;
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.subsample = subsample;
        this.seed = seed;
    }

    @Override
    public GradientBoostingClassifier fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();

        // Extract classes
        Set<Integer> clsSet = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            clsSet.add((int) y.get(i));
        }
        nClasses = clsSet.size();
        classes = new int[nClasses];
        int ci = 0;
        for (int c : clsSet) {
            classes[ci++] = c;
        }

        // Encode labels 0..K-1
        int[] yEncoded = new int[n];
        Map<Integer, Integer> labelMap = new HashMap<>();
        for (int i = 0; i < nClasses; i++) {
            labelMap.put(classes[i], i);
        }
        for (int i = 0; i < n; i++) {
            yEncoded[i] = labelMap.get((int) y.get(i));
        }

        // One-hot targets
        double[][] yArr = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            yArr[i][yEncoded[i]] = 1.0;
        }

        // Initial predictions: log priors
        initPredictions = new double[nClasses];
        int[] classCounts = new int[nClasses];
        for (int i = 0; i < n; i++) {
            classCounts[yEncoded[i]]++;
        }
        for (int k = 0; k < nClasses; k++) {
            double p = (double) classCounts[k] / n;
            p = Math.max(p, 1e-15);
            if (nClasses <= 2) {
                initPredictions[k] = Math.log(p / (1.0 - p));
            } else {
                initPredictions[k] = Math.log(p);
            }
        }

        // rawPred: F_k(x_i) for each sample i and class k
        double[][] rawPred = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            System.arraycopy(initPredictions, 0, rawPred[i], 0, nClasses);
        }

        estimators = new ArrayList<>();
        Random rng = new Random(seed);

        for (int iter = 0; iter < nEstimators; iter++) {
            boolean[] sampleMask = new boolean[n];
            if (subsample < 1.0) {
                int subSize = Math.max(1, (int) Math.round(n * subsample));
                List<Integer> pool = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    pool.add(i);
                }
                Collections.shuffle(pool, rng);
                for (int i = 0; i < subSize; i++) {
                    sampleMask[pool.get(i)] = true;
                }
            } else {
                Arrays.fill(sampleMask, true);
            }

            List<DecisionTreeRegressor> stageTrees = new ArrayList<>();
            for (int k = 0; k < nClasses; k++) {
                double[] negGrad = new double[n];
                for (int i = 0; i < n; i++) {
                    double p;
                    if (nClasses <= 2) {
                        p = 1.0 / (1.0 + Math.exp(-rawPred[i][k]));
                    } else {
                        double sumExp = 0;
                        for (int j = 0; j < nClasses; j++) {
                            sumExp += Math.exp(rawPred[i][j]);
                        }
                        p = Math.exp(rawPred[i][k]) / sumExp;
                    }
                    negGrad[i] = yArr[i][k] - p;
                }

                // Fit regression tree to negative gradient
                double[][] subData = new double[n][m];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        subData[i][j] = X.get(i, j);
                    }
                }

                int subN = 0;
                for (boolean b : sampleMask) {
                    if (b) {
                        subN++;
                    }
                }

                double[][] fitData = new double[subN][m];
                double[] fitTarget = new double[subN];
                int idx = 0;
                for (int i = 0; i < n; i++) {
                    if (sampleMask[i]) {
                        System.arraycopy(subData[i], 0, fitData[idx], 0, m);
                        fitTarget[idx] = negGrad[i];
                        idx++;
                    }
                }

                DecisionTreeRegressor tree = new DecisionTreeRegressor(
                    maxDepth, minSamplesSplit, minSamplesLeaf, false, seed + iter * nClasses + k);
                tree.fit(new Matrix(fitData), new Vector(fitTarget));

                // Predict on all data and update raw predictions
                Vector treePred = tree.predict(new Matrix(subData));
                for (int i = 0; i < n; i++) {
                    rawPred[i][k] += learningRate * treePred.get(i);
                }

                stageTrees.add(tree);
            }
            estimators.add(stageTrees);
        }
        return this;
    }

    @Override
    public Vector predict(Matrix X) {
        double[][] rawPred = predictRaw(X);
        int n = X.rows();
        Vector result = new Vector(n);
        for (int i = 0; i < n; i++) {
            int bestK = 0;
            for (int k = 1; k < nClasses; k++) {
                if (rawPred[i][k] > rawPred[i][bestK]) {
                    bestK = k;
                }
            }
            result.set(i, classes[bestK]);
        }
        return result;
    }

    public Vector predictProba(Matrix X) {
        double[][] rawPred = predictRaw(X);
        int n = X.rows();
        Vector prob = new Vector(n);
        for (int i = 0; i < n; i++) {
            double p;
            if (nClasses <= 2) {
                p = 1.0 / (1.0 + Math.exp(-rawPred[i][0]));
            } else {
                double sumExp = 0;
                for (int k = 0; k < nClasses; k++) {
                    sumExp += Math.exp(rawPred[i][k]);
                }
                p = Math.exp(rawPred[i][1]) / sumExp;
            }
            prob.set(i, p);
        }
        return prob;
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
        p.put("loss", loss);
        p.put("learning_rate", learningRate);
        p.put("n_estimators", nEstimators);
        p.put("max_depth", maxDepth);
        p.put("min_samples_split", minSamplesSplit);
        p.put("min_samples_leaf", minSamplesLeaf);
        p.put("subsample", subsample);
        return p;
    }

    private double[][] predictRaw(Matrix X) {
        int n = X.rows();
        double[][] rawPred = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            System.arraycopy(initPredictions, 0, rawPred[i], 0, nClasses);
        }

        int m = X.cols();
        double[][] xData = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                xData[i][j] = X.get(i, j);
            }
        }
        Matrix xMat = new Matrix(xData);

        for (int t = 0; t < estimators.size(); t++) {
            for (int k = 0; k < nClasses; k++) {
                Vector pred = estimators.get(t).get(k).predict(xMat);
                for (int i = 0; i < n; i++) {
                    rawPred[i][k] += learningRate * pred.get(i);
                }
            }
        }
        return rawPred;
    }
}
