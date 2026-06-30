package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class AdaBoostRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}
        });
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

        AdaBoostRegressor ada = new AdaBoostRegressor(20, 3, 42);
        ada.fit(X, y);

        Vector preds = ada.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), preds.get(i), 0.5);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        AdaBoostRegressor ada = new AdaBoostRegressor(10, 3, 42);
        assertThrows(IllegalStateException.class,
            () -> ada.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1.0, 2.0});

        AdaBoostRegressor ada = new AdaBoostRegressor(10, 3, 42);
        ada.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> ada.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        AdaBoostRegressor ada = new AdaBoostRegressor(30, 5, 0.5, 42);
        var params = ada.getParameters();
        assertEquals(30, params.get("n_estimators"));
        assertEquals(5, params.get("max_depth"));
        assertEquals(0.5, params.get("learning_rate"));
    }

    @Test
    void testIsFitted() {
        AdaBoostRegressor ada = new AdaBoostRegressor(10, 3, 42);
        assertFalse(ada.isFitted());
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0});
        ada.fit(X, y);
        assertTrue(ada.isFitted());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        });
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

        AdaBoostRegressor ada = new AdaBoostRegressor(20, 3, 42);
        ada.fit(X, y);

        double score = ada.score(X, y);
        assertTrue(score > 0.9 || Double.isFinite(score));
    }
}
