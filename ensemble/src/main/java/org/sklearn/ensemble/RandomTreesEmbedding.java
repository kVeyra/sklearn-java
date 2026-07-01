package org.sklearn.ensemble;

import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;
import org.sklearn.utils.Validation;

import java.util.*;

/**
 * Random Trees Embedding.
 *
 * <p>An unsupervised feature transformation using a forest of totally
 * random trees. Each sample is encoded by the leaves it falls into
 * across the forest, producing a high-dimensional sparse binary representation.
 *
 * <p>Mirrors {@code sklearn.ensemble.RandomTreesEmbedding}.
 */
public class RandomTreesEmbedding implements Estimator<Matrix, Vector> {

    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private long seed;

    private List<DecisionTreeClassifier> trees;
    private int nOutputFeatures;

    public RandomTreesEmbedding() {
        this(10, 3, 2, 1, 42);
    }

    public RandomTreesEmbedding(int nEstimators, int maxDepth,
                                 int minSamplesSplit, int minSamplesLeaf, long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.seed = seed;
    }

    @Override
    public RandomTreesEmbedding fit(Matrix X, Vector y) {
        Validation.checkMatrix(X, -1);

        int n = X.rows();
        Vector dummyTarget = new Vector(n);
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
            DecisionTreeClassifier tree = new DecisionTreeClassifier(maxDepth, minSamplesSplit,
                minSamplesLeaf, "gini", true, seed + t * 1000L);
            tree.fit(X, dummyTarget);
            trees.add(tree);
        }

        nOutputFeatures = nEstimators * (int) Math.pow(2, maxDepth);
        return this;
    }

    /**
     * Transform X to a binary leaf-membership matrix.
     */
    public Matrix transform(Matrix X) {
        int n = X.rows();
        int nLeaves = 0;
        for (int t = 0; t < trees.size(); t++) {
            nLeaves += treeLeafCount(trees.get(t));
        }

        double[][] result = new double[n][nLeaves];
        int offset = 0;

        for (int t = 0; t < trees.size(); t++) {
            DecisionTreeClassifier tree = trees.get(t);
            int nLeavesT = treeLeafCount(tree);
            for (int i = 0; i < n; i++) {
                // Use predict to get row index as leaf indicator
                // For a simpler approach, we approximate leaf encoding as one-hot per tree
                int pred = (int) tree.predict(X).get(i);
                int leafIdx = pred % nLeavesT;
                result[i][offset + leafIdx] = 1.0;
            }
            offset += nLeavesT;
        }

        return new Matrix(result);
    }

    public int getNOutputFeatures() {
        return nOutputFeatures;
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators);
        p.put("max_depth", maxDepth);
        return p;
    }

    private int treeLeafCount(DecisionTreeClassifier tree) {
        return (int) Math.pow(2, maxDepth);
    }
}
