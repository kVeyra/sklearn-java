package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class IsolationForestTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {10, 11}, {11, 12}, {12, 13},
            {100, 200}
        });
        Vector y = new Vector(X.rows());

        IsolationForest iforest = new IsolationForest(50, 0.1, 42);
        iforest.fit(X, y);
        Vector pred = iforest.predict(X);
        assertEquals(X.rows(), pred.size());
    }

    @Test
    void testAnomalyDetection() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
            {100, 200}
        });
        Vector y = new Vector(X.rows());

        IsolationForest iforest = new IsolationForest(50, 0.3, 42);
        iforest.fit(X, y);
        Vector pred = iforest.predict(X);
        // All predictions should be -1 or 1
        assertEquals(6, pred.size());
        for (int i = 0; i < pred.size(); i++) {
            assertTrue(pred.get(i) == -1 || pred.get(i) == 1);
        }
    }

    @Test
    void testScoreSamples() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}
        });
        Vector y = new Vector(3);

        IsolationForest iforest = new IsolationForest(20, 0.1, 42);
        iforest.fit(X, y);
        Vector scores = iforest.scoreSamples(X);
        assertEquals(3, scores.size());
        for (int i = 0; i < scores.size(); i++) {
            assertTrue(scores.get(i) > 0);
        }
    }
}
