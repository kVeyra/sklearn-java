package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RidgeClassifierTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 1}, {2, 2}, {5, 5}, {6, 6}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        RidgeClassifier rc = new RidgeClassifier(1.0, true);
        rc.fit(X, y);
        Vector preds = rc.predict(X);
        assertEquals(4, preds.size());
        double acc = rc.score(X, y);
        assertTrue(acc > 0.5);
    }

    @Test
    void testParameters() {
        RidgeClassifier rc = new RidgeClassifier();
        assertTrue(rc.getParameters().containsKey("alpha"));
    }
}
