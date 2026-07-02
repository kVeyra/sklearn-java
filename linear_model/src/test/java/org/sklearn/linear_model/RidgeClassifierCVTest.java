package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RidgeClassifierCVTest {

    @Test
    void testCVClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {5, 5}, {6, 6}, {7, 7}, {8, 8}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        RidgeClassifierCV rcv = new RidgeClassifierCV(
            new double[]{0.1, 1.0, 10.0}, 3);
        rcv.fit(X, y);
        Vector preds = rcv.predict(X);
        assertEquals(6, preds.size());
        assertTrue(rcv.isFitted());
    }

    @Test
    void testAlphaSelected() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {10, 1}, {11, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        RidgeClassifierCV rcv = new RidgeClassifierCV();
        rcv.fit(X, y);
        assertTrue(Double.isFinite(rcv.getAlpha()));
    }
}
