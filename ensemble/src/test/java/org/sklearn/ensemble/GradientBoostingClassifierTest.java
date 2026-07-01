package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class GradientBoostingClassifierTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {10, 11}, {11, 12}, {12, 13}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        GradientBoostingClassifier gbc = new GradientBoostingClassifier(20, 2, 42);
        gbc.fit(X, y);
        Vector pred = gbc.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), pred.get(i), 0.5);
        }
        assertTrue(gbc.score(X, y) > 0.8);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        GradientBoostingClassifier gbc = new GradientBoostingClassifier(20, 2, 42);
        gbc.fit(X, y);
        Vector proba = gbc.predictProba(X);
        assertEquals(4, proba.size());
        for (int i = 0; i < proba.size(); i++) {
            assertTrue(proba.get(i) >= 0 && proba.get(i) <= 1);
        }
    }

    @Test
    void testGetParameters() {
        GradientBoostingClassifier gbc = new GradientBoostingClassifier();
        assertNotNull(gbc.getParameters());
        assertTrue(gbc.getParameters().containsKey("learning_rate"));
    }
}
