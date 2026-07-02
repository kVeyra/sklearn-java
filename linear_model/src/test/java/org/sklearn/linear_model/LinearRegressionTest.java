package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class LinearRegressionTest {

    @Test
    void testSimpleLinearFit() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        assertEquals(2.0, model.getCoef().get(0), 1e-10);
        assertEquals(0.0, model.getIntercept(), 1e-10);
    }

    @Test
    void testFitWithIntercept() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 4}, {3, 1}, {4, 3}});
        Vector y = new Vector(new double[]{5, 9, 8, 12});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        // y = 2*x0 + 1*x1 + 1
        // X = [[1,2],[2,4],[3,1],[4,3]], y = [5,9,8,12]
        assertEquals(2.0, model.getCoef().get(0), 1e-10);
        assertEquals(1.0, model.getCoef().get(1), 1e-10);
        assertEquals(1.0, model.getIntercept(), 1e-10);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{3, 5, 7, 9, 11});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        Vector preds = model.predict(new Matrix(new double[][]{{6}, {7}}));
        assertEquals(2, preds.size());
        assertEquals(13.0, preds.get(0), 1e-10);
        assertEquals(15.0, preds.get(1), 1e-10);
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);
        double score = model.score(X, y);

        assertEquals(1.0, score, 1e-10);
    }

    @Test
    void testNoIntercept() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}});
        Vector y = new Vector(new double[]{5, 8, 11});

        LinearRegression model = new LinearRegression(false);
        model.fit(X, y);

        assertEquals(0.0, model.getIntercept(), 1e-10);
        assertEquals(1.0, model.getCoef().get(0), 1e-10);
        assertEquals(2.0, model.getCoef().get(1), 1e-10);
    }

    @Test
    void testSingleFeature() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Vector y = new Vector(new double[]{2.0, 4.0, 6.0});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        Vector pred = model.predict(new Matrix(new double[][]{{4.0}}));
        assertEquals(8.0, pred.get(0), 1e-10);
    }

    @Test
    void testSingleSample() {
        Matrix X = new Matrix(new double[][]{{1.0, 2.0}});
        Vector y = new Vector(new double[]{5.0});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        assertNotNull(model.getCoef());
        assertEquals(2, model.getCoef().size());
    }

    @Test
    void testPredictBeforeFitThrows() {
        LinearRegression model = new LinearRegression();
        Matrix X = new Matrix(new double[][]{{1.0}});
        assertThrows(IllegalStateException.class, () -> model.predict(X));
    }

    @Test
    void testScoreBeforeFitThrows() {
        LinearRegression model = new LinearRegression();
        Matrix X = new Matrix(new double[][]{{1.0}});
        Vector y = new Vector(new double[]{1.0});
        assertThrows(IllegalStateException.class, () -> model.score(X, y));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1, 2});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> model.predict(Xtest));
    }

    @Test
    void testPerfectPrediction() {
        Matrix X = new Matrix(new double[][]{{0}, {1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{0, 2, 4, 6, 8});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        assertEquals(2.0, model.getCoef().get(0), 1e-10);
        assertEquals(0.0, model.getIntercept(), 1e-10);
        assertEquals(1.0, model.score(X, y), 1e-10);
    }

    @Test
    void testGetParameters() {
        LinearRegression model = new LinearRegression(false);
        Matrix X = new Matrix(new double[][]{{1}, {2}});
        Vector y = new Vector(new double[]{1, 2});
        model.fit(X, y);

        var params = model.getParameters();
        assertTrue(params.containsKey("fit_intercept"));
        assertTrue(params.containsKey("coef_"));
        assertTrue(params.containsKey("intercept_"));
        assertEquals(false, params.get("fit_intercept"));
    }

    @Test
    void testR2ScoreLessThanOne() {
        Matrix X = new Matrix(new double[][]{{1, 10}, {2, 4}, {3, 7}, {4, 2}});
        // y_clean = 2*x0 + 3*x1 - 4 = [28, 12, 23, 10]
        // y with noise = [27, 13, 22, 11]
        Vector y = new Vector(new double[]{27, 13, 22, 11});

        LinearRegression model = new LinearRegression();
        model.fit(X, y);

        double score = model.score(X, y);
        assertTrue(score > 0.9 && score <= 1.0);
    }
}
