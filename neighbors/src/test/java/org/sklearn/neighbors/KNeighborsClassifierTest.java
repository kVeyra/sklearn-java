package org.sklearn.neighbors;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class KNeighborsClassifierTest {

    @Test
    void testSimpleClassification() {
        Matrix X = new Matrix(new double[][]{
            {1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}
        });
        Vector y = new Vector(new double[]{0, 0, 0, 1, 1, 1});

        KNeighborsClassifier knn = new KNeighborsClassifier(3);
        knn.fit(X, y);

        assertEquals(1.0, knn.score(X, y), 0.01);
    }

    @Test
    void testPredict() {
        Matrix X = new Matrix(new double[][]{{1.0}, {2.0}, {10.0}, {11.0}});
        Vector y = new Vector(new double[]{0, 0, 1, 1});

        KNeighborsClassifier knn = new KNeighborsClassifier(3);
        knn.fit(X, y);

        Vector preds = knn.predict(new Matrix(new double[][]{{1.5}, {10.5}}));
        assertEquals(0.0, preds.get(0), 0.01);
        assertEquals(1.0, preds.get(1), 0.01);
    }

    @Test
    void testPredictBeforeFitThrows() {
        KNeighborsClassifier knn = new KNeighborsClassifier(3);
        assertThrows(IllegalStateException.class,
            () -> knn.predict(new Matrix(new double[][]{{1.0}})));
    }

    @Test
    void testFeatureMismatchThrows() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Vector y = new Vector(new double[]{0, 1});
        KNeighborsClassifier knn = new KNeighborsClassifier(3);
        knn.fit(X, y);
        assertThrows(IllegalArgumentException.class,
            () -> knn.predict(new Matrix(new double[][]{{1, 2, 3}})));
    }
}
