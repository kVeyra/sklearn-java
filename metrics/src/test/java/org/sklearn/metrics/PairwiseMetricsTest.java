package org.sklearn.metrics;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class PairwiseMetricsTest {

    @Test
    void testEuclideanDistances() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {3, 4}});
        Matrix D = PairwiseMetrics.euclideanDistances(X);
        assertEquals(2, D.rows());
        assertEquals(2, D.cols());
        assertEquals(0.0, D.get(0, 0), 1e-10);
        assertEquals(5.0, D.get(0, 1), 1e-10);
        assertEquals(5.0, D.get(1, 0), 1e-10);
        assertEquals(0.0, D.get(1, 1), 1e-10);
    }

    @Test
    void testEuclideanDistancesTwoMatrices() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}});
        Matrix Y = new Matrix(new double[][]{{0, 0}});
        Matrix D = PairwiseMetrics.euclideanDistances(X, Y);
        assertEquals(2, D.rows());
        assertEquals(1, D.cols());
        assertEquals(0.0, D.get(0, 0), 1e-10);
        assertTrue(D.get(1, 0) > 0);
    }

    @Test
    void testManhattanDistances() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {3, 4}});
        Matrix D = PairwiseMetrics.manhattanDistances(X);
        assertEquals(0.0, D.get(0, 0), 1e-10);
        assertEquals(7.0, D.get(0, 1), 1e-10);
        assertEquals(7.0, D.get(1, 0), 1e-10);
        assertEquals(0.0, D.get(1, 1), 1e-10);
    }

    @Test
    void testCosineSimilarity() {
        Matrix X = new Matrix(new double[][]{{1, 0}, {0, 1}});
        Matrix S = PairwiseMetrics.cosineSimilarity(X);
        assertEquals(1.0, S.get(0, 0), 1e-10);
        assertEquals(0.0, S.get(0, 1), 1e-10);
        assertEquals(0.0, S.get(1, 0), 1e-10);
        assertEquals(1.0, S.get(1, 1), 1e-10);
    }

    @Test
    void testPairwiseDistances() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {3, 4}});
        Matrix D = PairwiseMetrics.pairwiseDistances(X, X);
        assertEquals(2, D.rows());
        assertEquals(2, D.cols());
    }

    @Test
    void testRBFKernel() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}});
        Matrix K = PairwiseMetrics.rbfKernel(X, X, 0.5);
        assertEquals(1.0, K.get(0, 0), 1e-10);
        assertTrue(K.get(0, 1) > 0);
        assertTrue(K.get(0, 1) < 1);
    }

    @Test
    void testRBFKernelBetweenMatrices() {
        Matrix X = new Matrix(new double[][]{{0, 0}});
        Matrix Y = new Matrix(new double[][]{{1, 1}});
        Matrix K = PairwiseMetrics.rbfKernel(X, Y, 0.5);
        assertEquals(1, K.rows());
        assertEquals(1, K.cols());
        assertTrue(K.get(0, 0) > 0);
    }

    @Test
    void testLinearKernel() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix K = PairwiseMetrics.linearKernel(X, X);
        assertEquals(2, K.rows());
        assertEquals(2, K.cols());
    }

    @Test
    void testPolynomialKernel() {
        Matrix X = new Matrix(new double[][]{{1, 2}});
        Matrix K = PairwiseMetrics.polynomialKernel(X, X, 3.0, 1.0, 1.0);
        assertEquals(1, K.rows());
        assertEquals(1, K.cols());
        assertTrue(K.get(0, 0) > 0);
    }

    @Test
    void testSigmoidKernel() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}});
        Matrix K = PairwiseMetrics.sigmoidKernel(X, 1.0, 0.0);
        assertEquals(2, K.rows());
        assertEquals(2, K.cols());
    }

    @Test
    void testPairwiseKernels() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}});
        Matrix K = PairwiseMetrics.pairwiseKernels(X, X, "rbf", 0.5, 3.0, 1.0);
        assertEquals(2, K.rows());
        assertEquals(2, K.cols());
    }

    @Test
    void testNanEuclideanDistances() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {3, 4}});
        Matrix D = PairwiseMetrics.nanEuclideanDistances(X, X);
        assertEquals(0.0, D.get(0, 0), 1e-10);
        assertEquals(5.0, D.get(0, 1), 1e-10);
    }
}
