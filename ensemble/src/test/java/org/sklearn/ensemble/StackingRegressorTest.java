package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StackingRegressorTest {

    @Test
    void testRegression() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeRegressor(3, 2, 1));
        estimators.add(new DecisionTreeRegressor(2, 2, 1));

        StackingRegressor sr = new StackingRegressor(estimators,
            new DecisionTreeRegressor(3, 2, 1), 2);
        sr.fit(X, y);
        Vector pred = sr.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeRegressor(3, 2, 1));

        StackingRegressor sr = new StackingRegressor(estimators,
            new DecisionTreeRegressor(3, 2, 1), 2);
        sr.fit(X, y);
        assertTrue(sr.score(X, y) > 0);
    }
}
