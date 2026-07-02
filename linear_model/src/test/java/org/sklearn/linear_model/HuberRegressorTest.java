package org.sklearn.linear_model;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class HuberRegressorTest {

    @Test
    void testFitAndPredict() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {5}});
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});
        HuberRegressor hr = new HuberRegressor();
        hr.fit(X, y);
        Vector pred = hr.predict(new Matrix(new double[][]{{6}}));
        assertEquals(1, pred.size());
        assertFalse(Double.isNaN(pred.get(0)));
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        Vector y = new Vector(new double[]{1, 2, 3, 4});
        HuberRegressor hr = new HuberRegressor();
        hr.fit(X, y);
        double s = hr.score(X, y);
        assertFalse(Double.isNaN(s));
    }

    @Test
    void testOutliers() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}, {100}});
        Vector y = new Vector(new double[]{1, 2, 3, 4, 200});
        HuberRegressor hr = new HuberRegressor();
        hr.fit(X, y);
        assertNotNull(hr.getOutliers());
    }

    @Test
    void testPredictBeforeFitThrows() {
        HuberRegressor hr = new HuberRegressor();
        assertThrows(IllegalStateException.class, () -> hr.predict(new Matrix(1, 1)));
    }
}
