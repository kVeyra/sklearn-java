package org.sklearn.tree;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class DecisionTreeRegressorTest {

    @Test
    void testSimpleRegression() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        });
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});

        DecisionTreeRegressor reg = new DecisionTreeRegressor(5, 2, 1);
        reg.fit(X, y);

        Vector preds = reg.predict(X);
        for (int i = 0; i < y.size(); i++) {
            assertEquals(y.get(i), preds.get(i), 0.01);
        }
    }

    @Test
    void testConstantValue() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}
        });
        Vector y = new Vector(new double[]{5, 5, 5, 5});

        DecisionTreeRegressor reg = new DecisionTreeRegressor(5, 2, 1);
        reg.fit(X, y);

        Vector preds = reg.predict(new Matrix(new double[][]{{10.0}}));
        assertEquals(5.0, preds.get(0), 1e-10);
    }

    @Test
    void testStepFunction() {
        // y = 0 for x < 5, y = 1 for x >= 5
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {6.0}, {7.0}, {8.0}, {9.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 0, 1, 1, 1, 1});

        DecisionTreeRegressor reg = new DecisionTreeRegressor(3, 2, 1);
        reg.fit(X, y);

        assertEquals(0.0, reg.predict(new Matrix(new double[][]{{2.0}})).get(0), 0.01);
        assertEquals(1.0, reg.predict(new Matrix(new double[][]{{8.0}})).get(0), 0.01);
    }

    @Test
    void testMaxDepth() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 100, 110, 120});

        DecisionTreeRegressor shallow = new DecisionTreeRegressor(1, 2, 1);
        shallow.fit(X, y);

        DecisionTreeRegressor deep = new DecisionTreeRegressor(10, 2, 1);
        deep.fit(X, y);

        assertTrue(deep.score(X, y) >= shallow.score(X, y));
    }

    @Test
    void testR2Score() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        });
        Vector y = new Vector(new double[]{2, 4, 6, 8, 10});

        DecisionTreeRegressor reg = new DecisionTreeRegressor(5, 2, 1);
        reg.fit(X, y);

        assertEquals(1.0, reg.score(X, y), 0.01);
    }

    @Test
    void testPredictBeforeFitThrows() {
        DecisionTreeRegressor reg = new DecisionTreeRegressor(3, 2, 1);
        assertThrows(IllegalStateException.class,
            () -> reg.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{1, 2});

        DecisionTreeRegressor reg = new DecisionTreeRegressor(3, 2, 1);
        reg.fit(X, y);

        Matrix Xtest = new Matrix(new double[][]{{1, 2, 3}});
        assertThrows(IllegalArgumentException.class, () -> reg.predict(Xtest));
    }

    @Test
    void testGetParameters() {
        DecisionTreeRegressor reg = new DecisionTreeRegressor(5, 4, 2);
        var params = reg.getParameters();
        assertEquals(5, params.get("max_depth"));
        assertEquals(4, params.get("min_samples_split"));
        assertEquals(2, params.get("min_samples_leaf"));
    }
}
