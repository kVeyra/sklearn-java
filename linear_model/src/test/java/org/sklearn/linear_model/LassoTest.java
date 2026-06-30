package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class LassoTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        Lasso model = new Lasso(1e-10, true, 1e-8, 10000);
        model.fit(X, y);

        double score = model.score(X, y);
        assertTrue(score > 0.9999);
    }

    @Test
    void testShrinkage() {
        Matrix X = new Matrix(new double[][]{
            {1, 5}, {2, 6}, {3, 7}, {4, 8}
        });
        Vector y = new Vector(new double[]{2, 4, 6, 8});

        Lasso modelWeak = new Lasso(0.01);
        modelWeak.fit(X, y);

        Lasso modelStrong = new Lasso(5.0);
        modelStrong.fit(X, y);

        double l2Weak = 0;
        double l2Strong = 0;
        for (int j = 0; j < modelWeak.getCoef().size(); j++) {
            l2Weak += modelWeak.getCoef().get(j) * modelWeak.getCoef().get(j);
            l2Strong += modelStrong.getCoef().get(j) * modelStrong.getCoef().get(j);
        }

        assertTrue(l2Strong < l2Weak);
    }

    @Test
    void testNoIntercept() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}});
        Vector y = new Vector(new double[]{5, 8, 11});

        Lasso model = new Lasso(0.0, false, 1e-6, 5000);
        model.fit(X, y);

        assertEquals(0.0, model.getIntercept(), 1e-10);
    }

    @Test
    void testPredictBeforeFitThrows() {
        Lasso model = new Lasso();
        assertThrows(IllegalStateException.class,
            () -> model.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testNegativeAlphaThrows() {
        assertThrows(IllegalArgumentException.class, () -> new Lasso(-1.0));
    }

    @Test
    void testGetParameters() {
        Lasso model = new Lasso(0.1, false, 1e-3, 500);
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{1, 2});
        model.fit(X, y);

        var params = model.getParameters();
        assertEquals(0.1, params.get("alpha"));
        assertEquals(false, params.get("fit_intercept"));
    }
}
