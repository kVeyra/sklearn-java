package org.sklearn.multiclass;

import org.junit.jupiter.api.Test;
import org.sklearn.linear_model.LogisticRegression;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class OneVsRestClassifierTest {

    @Test
    void testFitPredict() {
        Matrix X = new Matrix(new double[][]{
            {1, 1},
            {2, 2},
            {3, 3},
            {10, 10},
            {11, 11}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1});

        OneVsRestClassifier ovr = new OneVsRestClassifier(new LogisticRegression());
        ovr.fit(X, y);
        Vector preds = ovr.predict(X);

        assertEquals(5, preds.size());
        assertEquals(0.0, preds.get(0));
        assertEquals(1.0, preds.get(3));
    }

    @Test
    void testThreeClasses() {
        Matrix X = new Matrix(new double[][]{
            {1, 0}, {2, 0}, {3, 0},
            {0, 1}, {0, 2}, {0, 3},
            {1, 1}, {2, 2}, {3, 3}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1, 2, 2, 2});

        OneVsRestClassifier ovr = new OneVsRestClassifier(new LogisticRegression());
        ovr.fit(X, y);
        Vector preds = ovr.predict(X);

        assertEquals(9, preds.size());
        assertEquals(3, ovr.getClasses().length);
    }

    @Test
    void testDecisionFunction() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {10, 10}});
        Vector y = new Vector(new double[]{0, 1});

        OneVsRestClassifier ovr = new OneVsRestClassifier(new LogisticRegression());
        ovr.fit(X, y);
        Matrix scores = ovr.decisionFunction(X);
        assertEquals(2, scores.rows());
        assertEquals(2, scores.cols());
    }
}
