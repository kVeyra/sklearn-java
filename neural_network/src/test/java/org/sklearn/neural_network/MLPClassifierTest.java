package org.sklearn.neural_network;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class MLPClassifierTest {

    @Test
    void testXorClassification() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        Vector y = new Vector(new double[]{0, 1, 1, 0});

        MLPClassifier mlp = new MLPClassifier(
            new int[]{4}, "relu", "adam", 0.0001, "constant", 0.01,
            500, true, 1e-5, 4, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        Vector preds = mlp.predict(X);
        assertEquals(0, preds.get(0), 0.01);
        assertEquals(1, preds.get(1), 0.01);
        assertEquals(1, preds.get(2), 0.01);
        assertEquals(0, preds.get(3), 0.01);
    }

    @Test
    void testFitPredictShape() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        MLPClassifier mlp = new MLPClassifier(
            new int[]{3}, "tanh", "sgd", 0.0001, "constant", 0.01,
            200, true, 1e-5, 4, false, 0.1, 10, 1
        );
        mlp.fit(X, y);
        Vector preds = mlp.predict(X);
        assertEquals(4, preds.size());
        for (int i = 0; i < 4; i++) {
            assertTrue(preds.get(i) == 0 || preds.get(i) == 1);
        }
        assertTrue(mlp.isFitted());
    }

    @Test
    void testPredictProbas() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {1, 1}
        });
        Vector y = new Vector(new double[]{0, 1});

        MLPClassifier mlp = new MLPClassifier(
            new int[]{2}, "logistic", "adam", 0.001, "constant", 0.01,
            200, true, 1e-5, 2, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        double[] proba = mlp.predictProbas(X);
        assertEquals(4, proba.length);
        for (int i = 0; i < proba.length; i++) {
            assertTrue(proba[i] >= 0 && proba[i] <= 1);
        }
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}
        });
        Vector y = new Vector(new double[]{1, 1, 0, 0});

        MLPClassifier mlp = new MLPClassifier(
            new int[]{4}, "relu", "adam", 0.0001, "constant", 0.01,
            300, true, 1e-5, 4, false, 0.1, 10, 7
        );
        mlp.fit(X, y);
        double score = mlp.score(X, y);
        assertTrue(score >= 0 && score <= 1);
    }

    @Test
    void testLossCurve() {
        Matrix X = new Matrix(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        Vector y = new Vector(new double[]{0, 1, 1, 0});

        MLPClassifier mlp = new MLPClassifier(
            new int[]{4}, "relu", "adam", 0.0001, "constant", 0.01,
            100, true, 1e-5, 4, false, 0.1, 10, 42
        );
        mlp.fit(X, y);
        assertNotNull(mlp.getLossCurve());
        assertTrue(mlp.getLossCurve().size() > 0);
        assertNotNull(mlp.getClasses());
        assertEquals(2, mlp.getClasses().length);
    }

    @Test
    void testParameters() {
        MLPClassifier mlp = new MLPClassifier(
            new int[]{10, 5}, "relu", "adam", 0.01, "constant", 0.001,
            100, true, 1e-4, 100, true, 0.1, 10, 42
        );
        assertTrue(mlp.getParameters().containsKey("hidden_layer_sizes"));
        assertTrue(mlp.getParameters().containsKey("activation"));
        assertTrue(mlp.getParameters().containsKey("solver"));
    }
}
