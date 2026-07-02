package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Isolation Forest for anomaly detection.
 *
 * <p>Isolates observations by randomly selecting a feature and split value.
 * Anomalies are identified by shorter average path lengths in the isolation trees.
 *
 * <p>Mirrors {@code sklearn.ensemble.IsolationForest}.
 */
public class IsolationForest implements Estimator<Matrix, Vector> {

    private int nEstimators;
    private double contamination;
    private long seed;

    private List<IsolationTree> trees;
    private double threshold;
    private double offset;

    static class IsolationTree {
        static class Node {
            int feature;
            double threshold;
            int left;
            int right;
            int size;
            boolean isLeaf;
        }

        Node[] nodes;
        int nodeCount;

        IsolationTree(Matrix X, int[] sampleIdx, int start, int end, int heightLimit, Random rng) {
            int n = end - start;
            int maxNodes = 2 * (int) Math.pow(2, Math.min(heightLimit, 15));
            nodes = new Node[Math.max(1, maxNodes)];
            nodeCount = 0;
            build(X, sampleIdx, start, end, 0, heightLimit, rng);
        }

        int build(Matrix X, int[] sampleIdx, int start, int end,
                   int depth, int heightLimit, Random rng) {
            int n = end - start;
            int id = nodeCount++;
            if (id >= nodes.length) {
                nodes = Arrays.copyOf(nodes, nodes.length * 2);
            }
            nodes[id] = new Node();
            Node node = nodes[id];

            if (depth >= heightLimit || n <= 1) {
                node.isLeaf = true;
                node.size = n;
                return id;
            }

            int m = X.cols();
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
                node.isLeaf = true;
                node.size = n;
                return id;
            }

            double threshold = minVal + rng.nextDouble() * (maxVal - minVal);

            int leftCount = 0;
            int rightCount = 0;
            for (int i = start; i < end; i++) {
                if (X.get(sampleIdx[i], f) < threshold) {
                    leftCount++;
                } else {
                    rightCount++;
                }
            }

            if (leftCount == 0 || rightCount == 0) {
                node.isLeaf = true;
                node.size = n;
                return id;
            }

            int[] leftOrder = new int[leftCount];
            int[] rightOrder = new int[rightCount];
            int li = 0, ri = 0;
            for (int i = start; i < end; i++) {
                int idx = sampleIdx[i];
                if (X.get(idx, f) < threshold) {
                    leftOrder[li++] = idx;
                } else {
                    rightOrder[ri++] = idx;
                }
            }

            System.arraycopy(leftOrder, 0, sampleIdx, start, leftCount);
            System.arraycopy(rightOrder, 0, sampleIdx, start + leftCount, rightCount);

            node.feature = f;
            node.threshold = threshold;
            node.left = build(X, sampleIdx, start, start + leftCount, depth + 1, heightLimit, rng);
            node.right = build(X, sampleIdx, start + leftCount, end, depth + 1, heightLimit, rng);
            return id;
        }

        double pathLength(Matrix X, int row) {
            return pathLength(X, row, 0, 0);
        }

        private double pathLength(Matrix X, int row, int nodeId, int depth) {
            Node node = nodes[nodeId];
            if (node.isLeaf) {
                return depth + cFactor(node.size);
            }
            if (X.get(row, node.feature) < node.threshold) {
                return pathLength(X, row, node.left, depth + 1);
            } else {
                return pathLength(X, row, node.right, depth + 1);
            }
        }

        static double cFactor(int n) {
            if (n <= 1) {
                return 0;
            }
            if (n == 2) {
                return 1;
            }
            return 2.0 * (Math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n);
        }
    }

    public IsolationForest() {
        this(100, 0.1, 42);
    }

    public IsolationForest(int nEstimators, double contamination, long seed) {
        this.nEstimators = nEstimators;
        this.contamination = contamination;
        this.seed = seed;
    }

    @Override
    public IsolationForest fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);

        int n = X.rows();
        trees = new ArrayList<>();
        Random rng = new Random(seed);
        int heightLimit = (int) Math.ceil(Math.log(n) / Math.log(2));

        for (int t = 0; t < nEstimators; t++) {
            Random treeRng = new Random(seed + t * 1000L);
            int[] sampleIdx = new int[n];
            for (int i = 0; i < n; i++) {
                sampleIdx[i] = treeRng.nextInt(n);
            }
            IsolationTree tree = new IsolationTree(X, sampleIdx, 0, n, heightLimit, treeRng);
            trees.add(tree);
        }

        // Compute anomaly scores for training data to determine threshold
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            scores[i] = anomalyScore(X, i);
        }
        Arrays.sort(scores);
        int thresholdIdx = (int) Math.round((1.0 - contamination) * (n - 1));
        thresholdIdx = Math.min(Math.max(0, thresholdIdx), n - 1);
        threshold = scores[thresholdIdx];
        offset = 0.5;

        return this;
    }

    /**
     * Predict anomaly scores (-1 for anomalies, 1 for inliers).
     */
    public Vector predict(Matrix X) {
        int n = X.rows();
        Vector result = new Vector(n);
        for (int i = 0; i < n; i++) {
            double s = anomalyScore(X, i);
            double decision = s - offset;
            result.set(i, decision < 0 ? -1 : 1);
        }
        return result;
    }

    public Vector scoreSamples(Matrix X) {
        int n = X.rows();
        Vector result = new Vector(n);
        for (int i = 0; i < n; i++) {
            result.set(i, anomalyScore(X, i));
        }
        return result;
    }

    public double getThreshold() {
        return threshold;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators);
        p.put("contamination", contamination);
        return p;
    }

    private double anomalyScore(Matrix X, int row) {
        double avgPath = 0;
        for (IsolationTree tree : trees) {
            avgPath += tree.pathLength(X, row);
        }
        avgPath /= trees.size();
        double c = IsolationTree.cFactor(X.rows());
        return Math.pow(2, -avgPath / c);
    }
}
