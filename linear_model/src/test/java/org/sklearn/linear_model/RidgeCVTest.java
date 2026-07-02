package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RidgeCVTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});
        RidgeCV cv = new RidgeCV(new double[]{0.1, 1.0, 10.0}, true, 3);
        cv.fit(X, y);
        Vector pred = cv.predict(new Matrix(new double[][]{{3}}));
        assertEquals(1, pred.size());
        assertTrue(cv.score(X, y) > 0.5);
    }

    @Test
    void testBestAlpha() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        RidgeCV cv = new RidgeCV(new double[]{0.1, 1.0}, true, null);
        cv.fit(X, y);
        assertTrue(cv.getAlpha() > 0);
    }

    @Test
    void testPredictBeforeFitThrows() {
        RidgeCV cv = new RidgeCV();
        assertThrows(IllegalStateException.class, () -> cv.predict(new Matrix(1, 1)));
    }
}
