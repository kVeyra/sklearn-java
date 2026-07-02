package org.sklearn.metrics;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ClusteringMetricsTest {

    @Test
    void testSilhouetteScore() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1}, {10, 10}, {10.1, 10.1}
        });
        Vector labels = new Vector(new double[]{0, 0, 1, 1});
        double s = ClusteringMetrics.silhouetteScore(X, labels);
        assertTrue(s > 0.8);
    }

    @Test
    void testAdjustedRandScorePerfect() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 0, 1, 1});
        assertEquals(1.0, ClusteringMetrics.adjustedRandScore(a, b), 1e-10);
    }

    @Test
    void testAdjustedRandScoreRandom() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 1, 0, 1});
        double ari = ClusteringMetrics.adjustedRandScore(a, b);
        assertTrue(Math.abs(ari) < 0.5);
    }

    @Test
    void testMutualInfoScore() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 0, 1, 1});
        assertTrue(ClusteringMetrics.mutualInfoScore(a, b) > 0);
    }

    @Test
    void testNormalizedMutualInfoScorePerfect() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 0, 1, 1});
        assertEquals(1.0, ClusteringMetrics.normalizedMutualInfoScore(a, b), 1e-10);
    }

    @Test
    void testHomogeneityCompletenessVMeasure() {
        Vector a = new Vector(new double[]{0, 0, 1, 1, 2, 2});
        Vector b = new Vector(new double[]{0, 0, 1, 1, 2, 2});
        assertEquals(1.0, ClusteringMetrics.homogeneityScore(a, b), 1e-10);
        assertEquals(1.0, ClusteringMetrics.completenessScore(a, b), 1e-10);
        assertEquals(1.0, ClusteringMetrics.vMeasureScore(a, b), 1e-10);
    }

    @Test
    void testRandScore() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 0, 1, 1});
        assertEquals(1.0, ClusteringMetrics.randScore(a, b), 1e-10);
    }

    @Test
    void testFowlkesMallowsScore() {
        Vector a = new Vector(new double[]{0, 0, 1, 1});
        Vector b = new Vector(new double[]{0, 0, 1, 1});
        assertEquals(1.0, ClusteringMetrics.fowlkesMallowsScore(a, b), 1e-10);
    }

    @Test
    void testCalinskiHarabaszScore() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0.1, 0.1}, {10, 10}, {10.1, 10.1}
        });
        Vector labels = new Vector(new double[]{0, 0, 1, 1});
        double ch = ClusteringMetrics.calinskiHarabaszScore(X, labels);
        assertTrue(ch > 100);
    }

    @Test
    void testLengthMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () ->
            ClusteringMetrics.adjustedRandScore(new Vector(2), new Vector(3)));
    }
}
