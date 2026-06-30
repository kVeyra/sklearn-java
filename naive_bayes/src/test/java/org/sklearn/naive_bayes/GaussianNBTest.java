package org.sklearn.naive_bayes;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class GaussianNBTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {1.5, 1.0}, {1.2, 0.8},
            {5.0, 5.0}, {5.5, 5.0}, {5.2, 4.8}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        GaussianNB nb = new GaussianNB();
        nb.fit(X, y);

        assertEquals(1.0, nb.score(X, y), 0.01);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{{1.0}, {10.0}});
        Vector y = new Vector(new double[]{0, 1});

        GaussianNB nb = new GaussianNB();
        nb.fit(X, y);

        Vector preds = nb.predict(new Matrix(new double[][]{{1.2}, {9.5}}));
        assertEquals(0.0, preds.get(0), 0.01);
        assertEquals(1.0, preds.get(1), 0.01);
    }

    @Test
    void testPredictBeforeFitThrows() {
        GaussianNB nb = new GaussianNB();
        assertThrows(IllegalStateException.class,
            () -> nb.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        GaussianNB nb = new GaussianNB();
        nb.fit(X, y);

        assertThrows(IllegalArgumentException.class,
            () -> nb.predict(new Matrix(new double[][]{{1, 2, 3}})));
    }

    @Test
    void testGetParameters() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        Vector y = new Vector(new double[]{0, 1});

        GaussianNB nb = new GaussianNB();
        nb.fit(X, y);

        var params = nb.getParameters();
        assertTrue(params.containsKey("theta_"));
        assertTrue(params.containsKey("sigma_"));
    }
}
