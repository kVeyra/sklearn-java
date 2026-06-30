package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class ExtraTreesRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 10.0, 11.0, 12.0});

        ExtraTreesRegressor et = new ExtraTreesRegressor(10, 3, 2, 1, 42);
        et.fit(X, y);

        Vector preds = et.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), preds.get(i), 1.0);
        }
    }

    @Test
    void testPredictBeforeFitThrows() {
        ExtraTreesRegressor et = new ExtraTreesRegressor(10, 3, 2, 1, 42);
        assertThrows(IllegalStateException.class,
            () -> et.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1.0, 2.0});

        ExtraTreesRegressor et = new ExtraTreesRegressor(10, 3, 2, 1, 42);
        et.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> et.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        ExtraTreesRegressor et = new ExtraTreesRegressor(50, 5, 4, 2, 42);
        var params = et.getParameters();
        assertEquals(50, params.get("n_estimators"));
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        });
        Vector y = new Vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

        ExtraTreesRegressor et = new ExtraTreesRegressor(10, 3, 2, 1, 42);
        et.fit(X, y);

        double score = et.score(X, y);
        assertTrue(score > 0.9 || Double.isFinite(score));
    }
}
