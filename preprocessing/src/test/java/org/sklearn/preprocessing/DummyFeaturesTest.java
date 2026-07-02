package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class DummyFeaturesTest {

    @Test
    void testAddDummyFeature() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix result = DummyFeatures.addDummyFeature(X, 1.0);
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(1.0, result.get(1, 0));
        assertEquals(1.0, result.get(0, 1));
        assertEquals(2.0, result.get(0, 2));
    }

    @Test
    void testCustomValue() {
        Matrix X = new Matrix(new double[][]{{5}});
        Matrix result = DummyFeatures.addDummyFeature(X, 0.5);
        assertEquals(0.5, result.get(0, 0));
        assertEquals(5.0, result.get(0, 1));
    }
}
