package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class KNeighborsRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 100, 110, 120});

        KNeighborsRegressor knn = new KNeighborsRegressor(2);
        knn.fit(X, y);

        double score = knn.score(X, y);
        assertTrue(score > 0.9);
    }

    @Test
    void testPredictBeforeFitThrows() {
        KNeighborsRegressor knn = new KNeighborsRegressor(3);
        assertThrows(IllegalStateException.class,
            () -> knn.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1, 2});
        KNeighborsRegressor knn = new KNeighborsRegressor(3);
        knn.fit(X, y);
        assertThrows(IllegalArgumentException.class,
            () -> knn.predict(new Matrix(new double[][]{{1, 2, 3}})));
    }
}
