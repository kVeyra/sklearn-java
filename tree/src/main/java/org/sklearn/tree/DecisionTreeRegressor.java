package org.sklearn.tree;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * A decision tree regressor.
 *
 * <p>Builds a binary decision tree using mean squared error (MSE)
 * as the splitting criterion. The tree is grown depth-first
 * by finding the best split at each node through exhaustive search
 * over all features and all possible split points.
 *
 * <p>Mirrors {@code sklearn.tree.DecisionTreeRegressor}.
 *
 * <p>Usage:
 * <pre>{@code
 * DecisionTreeRegressor reg = new DecisionTreeRegressor(3, 2, 1);
 * reg.fit(X, y);
 * Vector preds = reg.predict(X_test);
 * }</pre>
 */
public class DecisionTreeRegressor implements Predictor<Matrix, Vector, Vector> {

    static class Node {
        int feature = -1;
        double threshold = 0.0;
        double impurity = 0.0;
        double value;
        int left = -1;
        int right = -1;
        boolean isLeaf = false;
    }

    private Node[] nodes;
    private int nodeCount;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private boolean useRandomSplit;
    private boolean fitted;
    private int nFeatures;
    private long seed;

    /**
     * Create a decision tree regressor with default parameters.
     */
    public DecisionTreeRegressor() {
        this(Integer.MAX_VALUE, 2, 1, false, 42);
    }

    /**
     * Create a decision tree regressor.
     *
     * @param maxDepth         maximum depth of the tree
     * @param minSamplesSplit  minimum samples required to split
     * @param minSamplesLeaf   minimum samples required at a leaf
     */
    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit, int minSamplesLeaf) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, false, 42);
    }

    /**
     * Create a decision tree regressor with full control.
     *
     * @param maxDepth         maximum depth
     * @param minSamplesSplit  min samples required to split
     * @param minSamplesLeaf   min samples required at a leaf
     * @param useRandomSplit   if true, pick random thresholds (for ExtraTrees)
     * @param seed             random seed
     */
    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit,
                                  int minSamplesLeaf, boolean useRandomSplit, long seed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.useRandomSplit = useRandomSplit;
        this.seed = seed;
    }

    @Override
    public DecisionTreeRegressor fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        int n = X.rows();
        int m = X.cols();
        this.nFeatures = m;

        int[] sampleIndex = new int[n];
        for (int i = 0; i < n; i++) {
            sampleIndex[i] = i;
        }

        nodes = new Node[Math.max(1, 2 * (int) Math.pow(2, Math.min(maxDepth, 15)))];
        nodeCount = 0;

        buildTree(X, y, sampleIndex, 0, n, 0, new RandomGenerator(seed));

        fitted = true;
        return this;
    }

    private int buildTree(Matrix X, Vector y, int[] sampleIdx, int start,
                          int end, int depth, RandomGenerator rng) {
        int nodeId = nodeCount++;
        if (nodeId >= nodes.length) {
            nodes = Arrays.copyOf(nodes, nodes.length * 2);
        }
        nodes[nodeId] = new Node();
        Node node = nodes[nodeId];

        int n = end - start;
        double sum = 0.0;
        double sumSq = 0.0;
        for (int i = start; i < end; i++) {
            double val = y.get(sampleIdx[i]);
            sum += val;
            sumSq += val * val;
        }
        double mean = sum / n;
        double mse = sumSq / n - mean * mean;

        node.impurity = mse;
        node.value = mean;

        boolean shouldStop = depth >= maxDepth || n < minSamplesSplit;
        if (shouldStop) {
            node.isLeaf = true;
            return nodeId;
        }

        BestSplit best = findBestSplit(X, y, sampleIdx, start, end, rng);

        if (best == null || best.improvement < -1e-15) {
            node.isLeaf = true;
            return nodeId;
        }

        node.feature = best.feature;
        node.threshold = best.threshold;

        int[] leftOrder = new int[n];
        int[] rightOrder = new int[n];
        int leftCount = 0, rightCount = 0;

        for (int i = start; i < end; i++) {
            int idx = sampleIdx[i];
            if (X.get(idx, best.feature) <= best.threshold) {
                leftOrder[leftCount++] = idx;
            } else {
                rightOrder[rightCount++] = idx;
            }
        }

        if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) {
            node.isLeaf = true;
            return nodeId;
        }

        System.arraycopy(leftOrder, 0, sampleIdx, start, leftCount);
        System.arraycopy(rightOrder, 0, sampleIdx, start + leftCount, rightCount);

        int leftChild = buildTree(X, y, sampleIdx, start, start + leftCount, depth + 1, rng);
        int rightChild = buildTree(X, y, sampleIdx, start + leftCount, end, depth + 1, rng);

        node.left = leftChild;
        node.right = rightChild;

        return nodeId;
    }

    static class BestSplit {
        int feature;
        double threshold;
        double improvement;
    }

    private BestSplit findBestSplit(Matrix X, Vector y, int[] sampleIdx,
                                    int start, int end, RandomGenerator rng) {
        int n = end - start;
        int m = nFeatures;
        BestSplit best = null;

        double totalMean = 0.0;
        for (int i = start; i < end; i++) {
            totalMean += y.get(sampleIdx[i]);
        }
        totalMean /= n;
        double totalVar = 0.0;
        for (int i = start; i < end; i++) {
            double diff = y.get(sampleIdx[i]) - totalMean;
            totalVar += diff * diff;
        }
        totalVar /= n;

        if (useRandomSplit) {
            int nAttempts = Math.min(m, 10);
            for (int attempt = 0; attempt < nAttempts; attempt++) {
                int f = rng.nextInt(m);
                double minVal = Double.POSITIVE_INFINITY;
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int i = start; i < end; i++) {
                    double v = X.get(sampleIdx[i], f);
                    if (v < minVal) {
                        minVal = v;
                    }
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
                if (minVal == maxVal) {
                    continue;
                }
                double threshold = minVal + rng.nextDouble() * (maxVal - minVal);

                int leftN = 0, rightN = 0;
                double leftSum = 0.0, leftSumSq = 0.0;
                double rightSum = 0.0, rightSumSq = 0.0;
                for (int i = start; i < end; i++) {
                    int idx = sampleIdx[i];
                    double val = y.get(idx);
                    if (X.get(idx, f) <= threshold) {
                        leftSum += val;
                        leftSumSq += val * val;
                        leftN++;
                    } else {
                        rightSum += val;
                        rightSumSq += val * val;
                        rightN++;
                    }
                }
                if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) {
                    continue;
                }

                double leftMean = leftSum / leftN;
                double leftMse = leftSumSq / leftN - leftMean * leftMean;
                double rightMean = rightSum / rightN;
                double rightMse = rightSumSq / rightN - rightMean * rightMean;

                double weightedChildMse =
                    (double) leftN / n * leftMse + (double) rightN / n * rightMse;
                double improvement = totalVar - weightedChildMse;

                if (improvement >= -1e-15 && (best == null || improvement > best.improvement)) {
                    best = new BestSplit();
                    best.feature = f;
                    best.threshold = threshold;
                    best.improvement = improvement;
                }
            }
            return best;
        }

        for (int f = 0; f < m; f++) {
            Integer[] sorted = new Integer[n];
            for (int i = 0; i < n; i++) {
                sorted[i] = i;
            }
            int fFinal = f;
            Arrays.sort(sorted, (a, b) -> {
                int ia = sampleIdx[start + a];
                int ib = sampleIdx[start + b];
                return Double.compare(X.get(ia, fFinal), X.get(ib, fFinal));
            });

            double leftSum = 0.0;
            double leftSumSq = 0.0;

            for (int s = 0; s < n - 1; s++) {
                int idx = sampleIdx[start + sorted[s]];
                double val = y.get(idx);
                leftSum += val;
                leftSumSq += val * val;
                double curVal = X.get(idx, f);
                double nextVal = X.get(sampleIdx[start + sorted[s + 1]], f);

                if (curVal == nextVal) {
                    continue;
                }

                int leftN = s + 1;
                int rightN = n - leftN;

                if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) {
                    continue;
                }

                double leftMean = leftSum / leftN;
                double leftMse = leftSumSq / leftN - leftMean * leftMean;

                double rightSum = totalMean * n - leftSum;
                double rightMean = rightSum / rightN;

                double rightMse = 0.0;
                if (rightN > 0) {
                    double rightSumSq = 0.0;
                    for (int i = s + 1; i < n; i++) {
                        int ri = sampleIdx[start + sorted[i]];
                        double rv = y.get(ri);
                        rightSumSq += rv * rv;
                    }
                    rightMse = rightSumSq / rightN - rightMean * rightMean;
                }

                double weightedChildMse =
                    (double) leftN / n * leftMse + (double) rightN / n * rightMse;
                double improvement = totalVar - weightedChildMse;
                double threshold = (curVal + nextVal) / 2.0;

                if (improvement >= -1e-15 && (best == null || improvement > best.improvement)) {
                    best = new BestSplit();
                    best.feature = f;
                    best.threshold = threshold;
                    best.improvement = improvement;
                }
            }
        }

        return best;
    }

    @Override
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "DecisionTreeRegressor");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            preds[i] = predictSample(X, i, 0);
        }
        return new Vector(preds);
    }

    private double predictSample(Matrix X, int row, int nodeId) {
        Node node = nodes[nodeId];
        if (node.isLeaf || node.left < 0) {
            return node.value;
        }
        if (X.get(row, node.feature) <= node.threshold) {
            return predictSample(X, row, node.left);
        } else {
            return predictSample(X, row, node.right);
        }
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "DecisionTreeRegressor");
        Validation.checkMatrix(X, -1);
        Validation.checkTarget(y, X.rows());

        Vector pred = predict(X);
        double ssRes = 0.0;
        double ssTot = 0.0;
        double yMean = y.mean();
        for (int i = 0; i < y.size(); i++) {
            double diff = y.get(i) - pred.get(i);
            ssRes += diff * diff;
            double diffMean = y.get(i) - yMean;
            ssTot += diffMean * diffMean;
        }
        if (ssTot == 0.0) {
            return 1.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    public boolean isFitted() {
        return fitted;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("max_depth", maxDepth);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        params.put("use_random_split", useRandomSplit);
        return Collections.unmodifiableMap(params);
    }
}
