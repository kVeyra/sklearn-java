package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class LogisticRegressionTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 1.0}, {2.0, 1.0}, {1.5, 0.5},
            {5.0, 4.0}, {6.0, 5.0}, {5.5, 4.5}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);

        Vector preds = model.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), preds.get(i), 0.5);
        }
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{{1.0}, {10.0}});
        Vector y = new Vector(new double[]{0, 1});

        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);

        Vector probs = model.predictProba(X);
        assertTrue(probs.get(0) < 0.5);
        assertTrue(probs.get(1) > 0.5);
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);

        double score = model.score(X, y);
        assertTrue(score >= 0.8);
    }

    @Test
    void testPerfectSeparation() {
        Matrix X = new Matrix(new double[][]{
            {0.0}, {0.1}, {0.2}, {10.0}, {10.1}, {10.2}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        LogisticRegression model = new LogisticRegression(10.0, true, 1e-6, 500);
        model.fit(X, y);

        assertEquals(1.0, model.score(X, y), 1e-6);
    }

    @Test
    void testPredictBeforeFitThrows() {
        LogisticRegression model = new LogisticRegression();
        Matrix X = new Matrix(new double[][]{{1.0}});
        assertThrows(IllegalStateException.class, () -> model.predict(X));
    }

    @Test
    void testScoreBeforeFitThrows() {
        LogisticRegression model = new LogisticRegression();
        Matrix X = new Matrix(new double[][]{{1.0}});
        Vector y = new Vector(new double[]{1.0});
        assertThrows(IllegalStateException.class, () -> model.score(X, y));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});

        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> model.predict(Xtest));
    }

    @Test
    void testSingleClassThrows() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Vector y = new Vector(new double[]{1, 1, 1});

        LogisticRegression model = new LogisticRegression();
        assertThrows(IllegalArgumentException.class, () -> model.fit(X, y));
    }

    @Test
    void testNegativeCThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> new LogisticRegression(-1.0, true, 1e-4, 100));
    }

    @Test
    void testNegativeTolThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> new LogisticRegression(1.0, true, -1.0, 100));
    }

    @Test
    void testNoIntercept() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 10.0}, {2.0, 20.0}, {1.0, 1.0}, {2.0, 2.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        LogisticRegression model = new LogisticRegression(1.0, false, 1e-4, 200);
        model.fit(X, y);

        assertEquals(0.0, model.getIntercept().get(0), 1e-10);
        assertNotNull(model.getCoef());
    }

    @Test
    void testGetParameters() {
        LogisticRegression model = new LogisticRegression(0.5, false, 1e-3, 50);
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}});
        Vector y = new Vector(new double[]{0, 1});
        model.fit(X, y);

        var params = model.getParameters();
        assertEquals(0.5, params.get("C"));
        assertEquals(false, params.get("fit_intercept"));
        assertEquals(1e-3, params.get("tol"));
    }

    @Test
    void testLargeRegularization() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0},
            {7.0}, {8.0}, {9.0}, {10.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1});

        // Strong regularization (small C) pushes weights toward zero
        LogisticRegression model = new LogisticRegression(0.01, true, 1e-5, 1000);
        model.fit(X, y);

        assertTrue(Math.abs(model.getCoef().get(0)) < 5.0);
    }
}
