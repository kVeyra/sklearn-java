package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class LocalOutlierFactorTest {

    @Test
    void testFitAndDetect() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}, {10, 10}, {11, 11}});
        LocalOutlierFactor lof = new LocalOutlierFactor(2);
        lof.fit(X);
        double[] scores = lof.getNegativeOutlierFactor();
        assertEquals(5, scores.length);
    }

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}, {10, 10}});
        LocalOutlierFactor lof = new LocalOutlierFactor(2);
        Vector scores = lof.fitPredict(X);
        assertEquals(4, scores.size());
    }

    @Test
    void testDecisionFunction() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}});
        LocalOutlierFactor lof = new LocalOutlierFactor(1);
        lof.fit(X);
        Vector scores = lof.decisionFunction(X);
        assertEquals(2, scores.size());
    }

    @Test
    void testNNeighborsFound() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}});
        LocalOutlierFactor lof = new LocalOutlierFactor(2);
        lof.fit(X);
        int[] found = lof.getNNeighborsFound();
        for (int f : found) {
            assertEquals(2, f);
        }
    }

    @Test
    void testFitBeforePredict() {
        LocalOutlierFactor lof = new LocalOutlierFactor();
        assertThrows(IllegalStateException.class, () -> lof.decisionFunction(new Matrix(1, 1)));
    }
}
