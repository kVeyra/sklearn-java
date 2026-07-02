package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RandomTreesEmbeddingTest {

    @Test
    void testFitAndTransform() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        RandomTreesEmbedding rte = new RandomTreesEmbedding(10, 3, 2, 1, 42);
        rte.fit(X, y);
        Matrix transformed = rte.transform(X);
        assertEquals(X.rows(), transformed.rows());
    }

    @Test
    void testGetNOutputFeatures() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}
        });
        Vector y = new Vector(new double[]{0, 0});

        RandomTreesEmbedding rte = new RandomTreesEmbedding(10, 3, 2, 1, 42);
        rte.fit(X, y);
        assertTrue(rte.getNOutputFeatures() > 0);
    }
}
