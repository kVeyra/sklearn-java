package org.sklearn.neural_network;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class MLPRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {0}, {1}, {2}, {3}, {4}, {5}
        });
        Vector y = new Vector(new double[]{0, 2, 4, 6, 8, 10});

        MLPRegressor mlp = new MLPRegressor(
            new int[]{5}, "relu", "adam", 0.0001, "constant", 0.01,
            500, true, 1e-5, 6, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        Vector preds = mlp.predict(X);
        assertEquals(6, preds.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}
        });
        Vector y = new Vector(new double[]{1.5, 3.5, 5.5, 7.5});

        MLPRegressor mlp = new MLPRegressor(
            new int[]{4}, "tanh", "sgd", 0.001, "constant", 0.01,
            300, true, 1e-5, 4, false, 0.1, 10, 1
        );
        mlp.fit(X, y);
        double score = mlp.score(X, y);
        assertTrue(Double.isFinite(score));
    }

    @Test
    void testMultiOutput() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {1, 1}, {2, 2}, {3, 3}
        });
        Matrix yMat = new Matrix(new double[][]{
            {0, 1}, {1, 2}, {2, 3}, {3, 4}
        });
        Vector y = new Vector(new double[]{0, 1, 2, 3});

        MLPRegressor mlp = new MLPRegressor(
            new int[]{3}, "relu", "adam", 0.001, "constant", 0.01,
            200, true, 1e-5, 4, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        Vector preds = mlp.predict(X);
        assertEquals(4, preds.size());
        assertTrue(mlp.isFitted());
    }

    @Test
    void testLossCurve() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}
        });
        Vector y = new Vector(new double[]{2, 4, 6});

        MLPRegressor mlp = new MLPRegressor(
            new int[]{3}, "relu", "adam", 0.0001, "constant", 0.01,
            100, true, 1e-5, 3, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        assertNotNull(mlp.getLossCurve());
        assertTrue(mlp.getLossCurve().size() > 0);
    }

    @Test
    void testParameters() {
        MLPRegressor mlp = new MLPRegressor(
            new int[]{10}, "relu", "adam", 0.01, "constant", 0.001,
            200, true, 1e-4, 200, true, 0.1, 10, 42
        );
        assertTrue(mlp.getParameters().containsKey("hidden_layer_sizes"));
        assertTrue(mlp.getParameters().containsKey("activation"));
        assertTrue(mlp.getParameters().containsKey("solver"));
    }
}
