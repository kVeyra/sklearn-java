package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ExtraTreeClassifierTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1});

        ExtraTreeClassifier etc = new ExtraTreeClassifier(3, 2, 1, "gini", 42);
        etc.fit(X, y);
        Vector pred = etc.predict(X);
        assertEquals(5, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        ExtraTreeClassifier etc = new ExtraTreeClassifier(3, 2, 1, "gini", 42);
        etc.fit(X, y);
        assertTrue(etc.score(X, y) > 0.5);
    }

    @Test
    void testPredictProba() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}
        });
        Vector y = new Vector(new double[]{0, 1});

        ExtraTreeClassifier etc = new ExtraTreeClassifier(3, 2, 1, "gini", 42);
        etc.fit(X, y);
        Vector proba = etc.predictProba(X);
        assertEquals(2, proba.size());
    }
}
