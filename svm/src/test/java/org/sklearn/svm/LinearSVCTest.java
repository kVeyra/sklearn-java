package org.sklearn.svm;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class LinearSVCTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {5, 5}, {6, 6}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        LinearSVC svc = new LinearSVC(1.0, 1e-4, 1000);
        svc.fit(X, y);
        Vector preds = svc.predict(X);
        assertEquals(4, preds.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {10, 1}, {11, 1}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        LinearSVC svc = new LinearSVC(1.0, 1e-4, 500);
        svc.fit(X, y);
        double score = svc.score(X, y);
        assertTrue(score > 0.5);
    }
}
