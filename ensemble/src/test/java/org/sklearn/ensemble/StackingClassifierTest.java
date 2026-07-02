package org.sklearn.ensemble;

import org.junit.jupiter.api.Test;
import org.sklearn.core.Estimator;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import org.sklearn.tree.DecisionTreeClassifier;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StackingClassifierTest {

    @Test
    void testBinaryClassification() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeClassifier(3, 2, 1));
        estimators.add(new DecisionTreeClassifier(2, 2, 1));

        Estimator<Matrix, Vector> finalEst = new DecisionTreeClassifier(3, 2, 1);
        StackingClassifier sc = new StackingClassifier(estimators, finalEst, 2);
        sc.fit(X, y);
        Vector pred = sc.predict(X);
        assertEquals(4, pred.size());
    }

    @Test
    void testScore() {
        Matrix X = new Matrix(new double[][]{
            {1, 2}, {2, 3}, {10, 11}, {11, 12}
        });
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        List<Estimator<Matrix, Vector>> estimators = new ArrayList<>();
        estimators.add(new DecisionTreeClassifier(3, 2, 1));

        StackingClassifier sc = new StackingClassifier(estimators,
            new DecisionTreeClassifier(3, 2, 1), 2);
        sc.fit(X, y);
        assertTrue(sc.score(X, y) > 0.5);
    }
}
