package org.sklearn.dummy;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class DummyRegressorTest {

    @Test
    void testMean() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{2, 4, 6, 8});

        DummyRegressor dr = new DummyRegressor("mean", null);
        dr.fit(X, y);
        Vector preds = dr.predict(X);
        for (int i = 0; i < 4; i++) {
            assertEquals(5.0, preds.get(i), 1e-12);
        }
    }

    @Test
    void testMedian() {
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{10, 20});

        DummyRegressor dr = new DummyRegressor("median", null);
        dr.fit(X, y);
        Vector preds = dr.predict(X);
        assertEquals(15.0, preds.get(0), 1e-12);
    }

    @Test
    void testConstant() {
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{0, 0});

        DummyRegressor dr = new DummyRegressor("constant", 42.0);
        dr.fit(X, y);
        Vector preds = dr.predict(X);
        assertEquals(42.0, preds.get(0), 1e-12);
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}});
        Vector y = new Vector(new double[]{5, 5, 5});

        DummyRegressor dr = new DummyRegressor("mean", null);
        dr.fit(X, y);
        double score = dr.score(X, y);
        assertEquals(1.0, score, 1e-12);
    }

    @Test
    void testParameters() {
        DummyRegressor dr = new DummyRegressor();
        assertTrue(dr.getParameters().containsKey("strategy"));
    }
}
