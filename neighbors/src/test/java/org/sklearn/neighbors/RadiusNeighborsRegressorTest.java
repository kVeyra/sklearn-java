package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RadiusNeighborsRegressorTest {

    @Test
    void testBasic() {
        Matrix X = new Matrix(new double[][]{{0}, {1}, {2}, {10}});
        Vector y = new Vector(new double[]{0, 1, 2, 10});
        RadiusNeighborsRegressor reg = new RadiusNeighborsRegressor(5.0, "uniform");
        reg.fit(X, y);
        assertTrue(reg.score(X, y) > 0);
    }

    @Test
    void testDistanceWeights() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {10, 10}});
        Vector y = new Vector(new double[]{1, 2, 3});
        RadiusNeighborsRegressor reg = new RadiusNeighborsRegressor(5.0, "distance");
        reg.fit(X, y);
        assertFalse(Double.isNaN(reg.score(X, y)));
    }

    @Test
    void testEmptyNeighborhood() {
        Matrix X = new Matrix(new double[][]{{0}, {10}});
        Vector y = new Vector(new double[]{1, 2});
        RadiusNeighborsRegressor reg = new RadiusNeighborsRegressor(1.0, "uniform");
        reg.fit(X, y);
        Vector pred = reg.predict(new Matrix(new double[][]{{100}}));
        assertEquals(0.0, pred.get(0), 0.001);
    }

    @Test
    void testPredictBeforeFitThrows() {
        RadiusNeighborsRegressor reg = new RadiusNeighborsRegressor();
        assertThrows(IllegalStateException.class, () -> reg.predict(new Matrix(1, 1)));
    }
}
