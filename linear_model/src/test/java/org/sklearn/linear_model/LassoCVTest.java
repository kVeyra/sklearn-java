package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class LassoCVTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});
        LassoCV cv = new LassoCV();
        cv.fit(X, y);
        Vector pred = cv.predict(new Matrix(new double[][]{{6}}));
        assertEquals(1, pred.size());
        assertTrue(cv.getAlpha() > 0);
    }

    @Test
    void testAlphasPath() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        LassoCV cv = new LassoCV(1e-3, 5, true, 500, 1e-4, 3, false, 42);
        cv.fit(X, y);
        assertNotNull(cv.getAlphas());
        assertNotNull(cv.getMsePath());
    }

    @Test
    void testPredictBeforeFitThrows() {
        LassoCV cv = new LassoCV();
        assertThrows(IllegalStateException.class, () -> {
            cv.predict(new Matrix(1, 1));
        });
    }
}
