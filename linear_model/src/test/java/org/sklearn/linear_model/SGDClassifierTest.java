package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class SGDClassifierTest {

    @Test
    void testBinary() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {3, 4}, {10, 11}, {11, 12}});
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1});
        SGDClassifier clf = new SGDClassifier("hinge", "l2", 0.0001, 0.15, true, 200, 1e-3, "constant", 0.01, 0.5, false, 0.1, 5, true, 42, false, false);
        clf.fit(X, y);
        Vector pred = clf.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), pred.get(i), 0.5);
        }
        assertTrue(clf.score(X, y) > 0.6);
    }

    @Test
    void testLogLoss() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {8, 9}, {9, 10}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});
        SGDClassifier clf = new SGDClassifier("log_loss", "l2", 0.0001, 0.15, true, 500, 1e-3, "invscaling", 0.01, 0.5, false, 0.1, 5, true, 42, false, false);
        clf.fit(X, y);
        Vector proba = clf.predictProba(X);
        assertEquals(4, proba.size());
        for (int i = 0; i < proba.size(); i++) {
            assertTrue(proba.get(i) >= 0 && proba.get(i) <= 1);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        SGDClassifier clf = new SGDClassifier();
        assertThrows(IllegalStateException.class, () -> clf.predict(new Matrix(1, 1)));
    }
}
