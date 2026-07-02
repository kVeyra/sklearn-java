package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class PassiveAggressiveClassifierTest {

    @Test
    void testBinary() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Vector y = new Vector(new double[]{0, 0, 0, 1});
        PassiveAggressiveClassifier pa = new PassiveAggressiveClassifier("hinge", 0.01, true, 1000, 1e-3, true, 42);
        pa.fit(X, y);
        Vector pred = pa.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {8, 9}, {9, 10}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        PassiveAggressiveClassifier pa = new PassiveAggressiveClassifier("hinge", 1.0, true, 500, 1e-3, true, 42);
        pa.fit(X, y);
        double s = pa.score(X, y);
        assertFalse(Double.isNaN(s));
    }

    @Test
    void testPredictBeforeFitThrows() {
        PassiveAggressiveClassifier pa = new PassiveAggressiveClassifier();
        assertThrows(IllegalStateException.class, () -> pa.predict(new Matrix(1, 1)));
    }
}
