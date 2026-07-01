package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeRegressor;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VotingRegressorTest {

    @Test
    void testRegression() {
        Matrix X = new Matrix(new double[][]{
            {1}, {2}, {3}, {4}
        });
        Vector y = new Vector(new double[]{1, 2, 3, 4});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeRegressor(2, 2, 1));
        estimators.add(new DecisionTreeRegressor(3, 2, 1));

        VotingRegressor vr = new VotingRegressor(estimators);
        vr.fit(X, y);
        Vector pred = vr.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{1, 2, 10, 11});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeRegressor(3, 2, 1));

        VotingRegressor vr = new VotingRegressor(estimators);
        vr.fit(X, y);
        assertTrue(vr.score(X, y) > 0);
    }
}
