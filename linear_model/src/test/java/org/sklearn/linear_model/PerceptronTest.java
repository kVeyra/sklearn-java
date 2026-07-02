package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class PerceptronTest {

    @Test
    void testBinary() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Vector y = new Vector(new double[]{0, 0, 0, 1});
        Perceptron p = new Perceptron("l2", 0.0001, true, 1000, 1e-3, true, 42);
        p.fit(X, y);
        Vector pred = p.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testClasses() {
        Perceptron p = new Perceptron();
        p.fit(new Matrix(new double[][]{{1}, {2}, {3}}), new Vector(new double[]{0, 0, 1}));
        assertNotNull(p.getClasses());
    }

    @Test
    void testPredictBeforeFitThrows() {
        Perceptron p = new Perceptron();
        assertThrows(IllegalStateException.class, () -> p.predict(new Matrix(1, 1)));
    }
}
