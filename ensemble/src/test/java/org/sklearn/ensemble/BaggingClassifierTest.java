package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;

import static org.junit.jupiter.api.Assertions.*;

class BaggingClassifierTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {10, 11}, {11, 12}, {12, 13}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        Estimator<Matrix, Vector> base = new DecisionTreeClassifier(3, 2, 1);
        BaggingClassifier bc = new BaggingClassifier(base, 10, 42);
        bc.fit(X, y);
        Vector pred = bc.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), pred.get(i), 0.5);
        }
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {8, 9}, {9, 10}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        BaggingClassifier bc = new BaggingClassifier(
            new DecisionTreeClassifier(3, 2, 1), 10, 42);
        bc.fit(X, y);
        assertTrue(bc.score(X, y) > 0.5);
    }

    @Test
    void testGetParameters() {
        BaggingClassifier bc = new BaggingClassifier(
            new DecisionTreeClassifier(3, 2, 1), 10, 42);
        assertNotNull(bc.getParameters());
    }
}
