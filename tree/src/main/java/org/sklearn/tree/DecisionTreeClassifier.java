package org.sklearn.tree;

import org.sklearn.core.Predictor;
import org.sklearn.math.Matrix;
import org.sklearn.math.RandomGenerator;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * A decision tree classifier.
 *
 * <p>Builds a binary decision tree using either Gini impurity or
 * entropy/information gain as the splitting criterion. The tree is grown
 * depth-first (recursive partitioning) by finding the best split at each
 * node through exhaustive search over all features and all possible
 * split points.
 *
 * <p>Mirrors {@code sklearn.tree.DecisionTreeClassifier}.
 *
 * <p>Usage:
 * <pre>{@code
 * DecisionTreeClassifier clf = new DecisionTreeClassifier(3, 2, 1);
 * clf.fit(X, y);
 * Vector preds = clf.predict(X_test);
 * Vector probs = clf.predictProba(X_test);
 * }</pre>
 */
public class DecisionTreeClassifier implements Predictor<Matrix, Vector, Vector> {

    /**
     * Internal tree node.
     */
    static class Node {
        int feature = -1;
        double threshold = 0.0;
        double impurity = 0.0;
        double value; // majority class
        double[] classCounts;
        int left = -1;
        int right = -1;
        boolean isLeaf = false;
    }

    private Node[] nodes;
    private int nodeCount;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private String criterion;
    private boolean useRandomSplit;
    private boolean fitted;
    private int[] classes;
    private int nFeatures;
    private long seed;

    /**
     * Create a decision tree classifier.
     *
     * @param maxDepth         maximum depth of the tree
     * @param minSamplesSplit  minimum samples required to split
     * @param minSamplesLeaf   minimum samples required at a leaf
     */
    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit, int minSamplesLeaf) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, "gini");
    }

    /**
     * Create a decision tree classifier.
     *
     * @param maxDepth         maximum depth of the tree
     * @param minSamplesSplit  minimum samples required to split
     * @param minSamplesLeaf   minimum samples required at a leaf
     * @param criterion        splitting criterion ("gini" or "entropy")
     */
    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit,
                                  int minSamplesLeaf, String criterion) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, criterion, false, 42);
    }

    /**
     * Create a decision tree classifier with full control.
     *
     * @param maxDepth         maximum depth
     * @param minSamplesSplit  min samples required to split
     * @param minSamplesLeaf   min samples required at a leaf
     * @param criterion        "gini" or "entropy"
     * @param useRandomSplit   if true, pick random thresholds (for ExtraTrees)
     * @param seed             random seed
     */
    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit,
                                  int minSamplesLeaf, String criterion,
                                  boolean useRandomSplit, long seed) {
        if (!criterion.equals("gini") && !criterion.equals("entropy")) {
            throw new IllegalArgumentException(
                "criterion must be 'gini' or 'entropy', got: " + criterion);
        }
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = criterion;
        this.useRandomSplit = useRandomSplit;
        this.seed = seed;
    }

    @Override
    public DecisionTreeClassifier fit(Matrix X, Vector y) {
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
        int[] sampleIndex = new int[n];
        for (int i = 0; i < n; i++) {
            sampleIndex[i] = i;
        }

        nodes = new Node[Math.max(1, 2 * (int) Math.pow(2, Math.min(maxDepth, 15)))];
        nodeCount = 0;

        buildTree(X, y, sampleIndex, 0, n, 0, nClasses, new RandomGenerator(seed));

        fitted = true;
        return this;
    }

    private int buildTree(Matrix X, Vector y, int[] sampleIdx, int start,
                          int end, int depth, int nClasses, RandomGenerator rng) {
        int nodeId = nodeCount++;
        if (nodeId >= nodes.length) {
            nodes = Arrays.copyOf(nodes, nodes.length * 2);
        }
        nodes[nodeId] = new Node();

        int n = end - start;
        double[] classCounts = new double[nClasses];
        for (int i = start; i < end; i++) {
            int label = (int) y.get(sampleIdx[i]);
            int classIdx = indexOf(classes, label);
            classCounts[classIdx]++;
        }

        Node node = nodes[nodeId];
        node.classCounts = classCounts;
        node.impurity = impurity(classCounts, n);

        int majorityClass = argmax(classCounts);
        node.value = classes[majorityClass];

        boolean shouldStop = depth >= maxDepth || n < minSamplesSplit;
        if (shouldStop) {
            node.isLeaf = true;
            return nodeId;
        }

        BestSplit best = findBestSplit(X, y, sampleIdx, start, end, nClasses, rng);

        if (best == null || best.improvement < -1e-15) {
            node.isLeaf = true;
            return nodeId;
        }

        node.feature = best.feature;
        node.threshold = best.threshold;

        // Partition samples
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

        // Copy partitioned order back
        System.arraycopy(leftOrder, 0, sampleIdx, start, leftCount);
        System.arraycopy(rightOrder, 0, sampleIdx, start + leftCount, rightCount);

        int leftChild = buildTree(X, y, sampleIdx, start, start + leftCount,
            depth + 1, nClasses, rng);
        int rightChild = buildTree(X, y, sampleIdx, start + leftCount, end,
            depth + 1, nClasses, rng);

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
                                    int start, int end, int nClasses,
                                    RandomGenerator rng) {
        int n = end - start;
        int m = nFeatures;
        BestSplit best = null;

        double[] totalCounts = new double[nClasses];
        for (int i = start; i < end; i++) {
            int label = (int) y.get(sampleIdx[i]);
            totalCounts[indexOf(classes, label)]++;
        }

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
                double[] leftCounts = new double[nClasses];
                for (int i = start; i < end; i++) {
                    int idx = sampleIdx[i];
                    if (X.get(idx, f) <= threshold) {
                        int label = (int) y.get(idx);
                        leftCounts[indexOf(classes, label)]++;
                        leftN++;
                    } else {
                        rightN++;
                    }
                }
                if (leftN < minSamplesLeaf || rightN < minSamplesLeaf) {
                    continue;
                }

                double leftImp = impurity(leftCounts, leftN);
                double rightImp = 0.0;
                if (rightN > 0) {
                    double[] rightCounts = new double[nClasses];
                    for (int k = 0; k < nClasses; k++) {
                        rightCounts[k] = totalCounts[k] - leftCounts[k];
                    }
                    rightImp = impurity(rightCounts, rightN);
                }

                double weightedChildImp =
                    (double) leftN / n * leftImp + (double) rightN / n * rightImp;
                double improvement = impurity(totalCounts, n) - weightedChildImp;

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

            double[] leftCounts = new double[nClasses];

            for (int s = 0; s < n - 1; s++) {
                int idx = sampleIdx[start + sorted[s]];
                int label = (int) y.get(idx);
                leftCounts[indexOf(classes, label)]++;
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

                double leftImp = impurity(leftCounts, leftN);
                double rightImp = 0.0;
                if (rightN > 0) {
                    double[] rightCounts = new double[nClasses];
                    for (int k = 0; k < nClasses; k++) {
                        rightCounts[k] = totalCounts[k] - leftCounts[k];
                    }
                    rightImp = impurity(rightCounts, rightN);
                }

                double weightedChildImp =
                    (double) leftN / n * leftImp + (double) rightN / n * rightImp;
                double improvement = impurity(totalCounts, n) - weightedChildImp;
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

    private double impurity(double[] counts, int n) {
        if (n == 0) {
            return 0.0;
        }
        if (criterion.equals("gini")) {
            double sumSq = 0.0;
            for (double c : counts) {
                double p = c / n;
                sumSq += p * p;
            }
            return 1.0 - sumSq;
        } else {
            double entropy = 0.0;
            for (double c : counts) {
                if (c > 0) {
                    double p = c / n;
                    entropy -= p * Math.log(p) / Math.log(2);
                }
            }
            return entropy;
        }
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

    private int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return i;
            }
        }
        return 0;
    }

    private double[] predictRaw(Matrix X) {
        int n = X.rows();
        double[] preds = new double[n];
        for (int i = 0; i < n; i++) {
            preds[i] = predictSample(X, i, 0);
        }
        return preds;
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
    public Vector predict(Matrix X) {
        Validation.checkFitted(fitted, "DecisionTreeClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }
        return new Vector(predictRaw(X));
    }

    /**
     * Predict class probabilities for samples in X.
     * Returns probability of the positive class (binary).
     */
    public Vector predictProba(Matrix X) {
        Validation.checkFitted(fitted, "DecisionTreeClassifier");
        Validation.checkMatrix(X, -1);
        if (X.cols() != nFeatures) {
            throw new IllegalArgumentException(
                "Feature dimension mismatch: expected " + nFeatures
                    + " features, got " + X.cols());
        }

        int n = X.rows();
        double[] probs = new double[n];
        for (int i = 0; i < n; i++) {
            Node node = getLeaf(X, i, 0);
            double total = 0;
            for (double c : node.classCounts) {
                total += c;
            }
            if (node.classCounts.length >= 2) {
                probs[i] = node.classCounts[1] / total;
            } else {
                probs[i] = 0.0;
            }
        }
        return new Vector(probs);
    }

    private Node getLeaf(Matrix X, int row, int nodeId) {
        Node node = nodes[nodeId];
        if (node.isLeaf || node.left < 0) {
            return node;
        }
        if (X.get(row, node.feature) <= node.threshold) {
            return getLeaf(X, row, node.left);
        } else {
            return getLeaf(X, row, node.right);
        }
    }

    @Override
    public double score(Matrix X, Vector y) {
        Validation.checkFitted(fitted, "DecisionTreeClassifier");
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

    public int[] getClasses() {
        return classes;
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
        params.put("criterion", criterion);
        params.put("use_random_split", useRandomSplit);
        return Collections.unmodifiableMap(params);
    }
}
