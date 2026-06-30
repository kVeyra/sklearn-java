package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RidgeTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        Ridge model = new Ridge(0.0);
        model.fit(X, y);

        assertEquals(1.0, model.score(X, y), 1e-10);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{3, 5, 7, 9, 11});

        Ridge model = new Ridge(0.0);
        model.fit(X, y);

        Vector preds = model.predict(new Matrix(new double[][]{{6}, {7}}));
        assertEquals(13.0, preds.get(0), 1e-10);
        assertEquals(15.0, preds.get(1), 1e-10);
    }

    @Test
    void testRegularization() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});

        Ridge modelStrong = new Ridge(10.0);
        modelStrong.fit(X, y);

        Ridge modelWeak = new Ridge(0.01);
        modelWeak.fit(X, y);

        // Stronger regularization should shrink coefficients more
        assertTrue(Math.abs(modelStrong.getCoef().get(0))
            < Math.abs(modelWeak.getCoef().get(0)));
    }

    @Test
    void testNoIntercept() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}});
        Vector y = new Vector(new double[]{5, 8, 11});

        Ridge model = new Ridge(0.0, false);
        model.fit(X, y);

        assertEquals(0.0, model.getIntercept(), 1e-10);
    }

    @Test
    void testPredictBeforeFitThrows() {
        Ridge model = new Ridge();
        assertThrows(IllegalStateException.class,
            () -> model.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testNegativeAlphaThrows() {
        assertThrows(IllegalArgumentException.class, () -> new Ridge(-1.0));
    }

    @Test
    void testGetParameters() {
        Ridge model = new Ridge(0.5, false);
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{1, 2});
        model.fit(X, y);

        var params = model.getParameters();
        assertEquals(0.5, params.get("alpha"));
        assertEquals(false, params.get("fit_intercept"));
    }
}
