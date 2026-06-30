package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class RandomForestRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 100, 110, 120});

        RandomForestRegressor rf = new RandomForestRegressor(10, 5, 2, 1);
        rf.fit(X, y);

        double score = rf.score(X, y);
        assertTrue(score > 0.9);
    }

    @Test
    void testPredictBeforeFitThrows() {
        RandomForestRegressor rf = new RandomForestRegressor(10, 3, 2, 1);
        assertThrows(IllegalStateException.class,
            () -> rf.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1, 2});

        RandomForestRegressor rf = new RandomForestRegressor(10, 3, 2, 1);
        rf.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> rf.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        RandomForestRegressor rf = new RandomForestRegressor(50, 5, 4, 2);
        var params = rf.getParameters();
        assertEquals(50, params.get("n_estimators"));
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
    }
}
