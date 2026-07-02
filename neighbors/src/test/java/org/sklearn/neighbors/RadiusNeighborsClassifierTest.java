package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;
import static org.junit.jupiter.api.Assertions.*;

class RadiusNeighborsClassifierTest {

    @Test
    void testBasic() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}, {10, 10}, {11, 11}});
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1});
        RadiusNeighborsClassifier clf = new RadiusNeighborsClassifier(5.0, "uniform", false, -1.0);
        clf.fit(X, y);
        assertTrue(clf.score(X, y) > 0);
    }

    @Test
    void testOutlierLabel() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {10, 10}});
        Vector y = new Vector(new double[]{0, 1});
        RadiusNeighborsClassifier clf = new RadiusNeighborsClassifier(1.0, "uniform", true, -1.0);
        clf.fit(X, y);
        Vector pred = clf.predict(new Matrix(new double[][]{{100, 100}}));
        assertEquals(-1.0, pred.get(0), 0.001);
    }

    @Test
    void testDistanceWeights() {
        Matrix X = new Matrix(new double[][]{{0, 0}, {1, 1}, {2, 2}});
        Vector y = new Vector(new double[]{0, 0, 1});
        RadiusNeighborsClassifier clf = new RadiusNeighborsClassifier(5.0, "distance", false, -1.0);
        clf.fit(X, y);
        assertNotNull(clf.getClasses());
    }

    @Test
    void testPredictBeforeFitThrows() {
        RadiusNeighborsClassifier clf = new RadiusNeighborsClassifier();
        assertThrows(IllegalStateException.class, () -> clf.predict(new Matrix(1, 1)));
    }
}
