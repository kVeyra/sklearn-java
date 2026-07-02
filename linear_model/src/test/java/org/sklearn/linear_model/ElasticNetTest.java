package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ElasticNetTest {

    @Test
    void testBasicFit() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        ElasticNet model = new ElasticNet(1e-10, 0.5, true, 1e-8, 10000);
        model.fit(X, y);

        double score = model.score(X, y);
        assertTrue(score > 0.9999);
    }

    @Test
    void testCombinedPenalty() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        ElasticNet modelL2 = new ElasticNet(0.1, 0.0, true, 1e-6, 5000);
        modelL2.fit(X, y);

        ElasticNet modelL1 = new ElasticNet(0.1, 1.0, true, 1e-6, 5000);
        modelL1.fit(X, y);

        assertTrue(modelL2.score(X, y) > 0.9);
        assertTrue(modelL1.score(X, y) > 0.9);
    }

    @Test
    void testNoIntercept() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}});
        Vector y = new Vector(new double[]{5, 8, 11});

        ElasticNet model = new ElasticNet(0.0, 0.5, false, 1e-6, 5000);
        model.fit(X, y);

        assertEquals(0.0, model.getIntercept(), 1e-10);
    }

    @Test
    void testPredictBeforeFitThrows() {
        ElasticNet model = new ElasticNet();
        assertThrows(IllegalStateException.class,
            () -> model.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testInvalidAlphaThrows() {
        assertThrows(IllegalArgumentException.class, () -> new ElasticNet(-1.0, 0.5));
    }

    @Test
    void testInvalidL1RatioThrows() {
        assertThrows(IllegalArgumentException.class, () -> new ElasticNet(1.0, 1.5));
    }

    @Test
    void testL2EquivalentToLasso() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        ElasticNet enetL1 = new ElasticNet(0.1, 1.0);
        enetL1.fit(X, y);

        Lasso lasso = new Lasso(0.1);
        lasso.fit(X, y);

        double scoreDiff = Math.abs(enetL1.score(X, y) - lasso.score(X, y));
        assertTrue(scoreDiff < 1e-4);
    }

    @Test
    void testGetParameters() {
        ElasticNet model = new ElasticNet(0.1, 0.3, false, 1e-3, 500);
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{1, 2});
        model.fit(X, y);

        var params = model.getParameters();
        assertEquals(0.1, params.get("alpha"));
        assertEquals(0.3, params.get("l1_ratio"));
    }
}
