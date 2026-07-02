package org.sklearn.model_selection;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class TrainTestSplitTest {

    @Test
    void testSplitShapes() {
        Matrix X = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});
        Vector y = new Vector(new double[]{0, 1, 0, 1, 0});

        var split = TrainTestSplit.split(X, y, 0.2, 42);
        assertEquals(4, split.xTrain.rows());
        assertEquals(1, split.xTest.rows());
        assertEquals(4, split.yTrain.size());
        assertEquals(1, split.yTest.size());
        assertEquals(2, split.xTrain.cols());
    }

    @Test
    void testDeterministic() {
        Matrix X = new Matrix(new double[][]{{1}, {2}, {3}, {4}});
        Vector y = new Vector(new double[]{0, 1, 0, 1});

        var s1 = TrainTestSplit.split(X, y, 0.25, 42);
        var s2 = TrainTestSplit.split(X, y, 0.25, 42);
        for (int i = 0; i < s1.yTrain.size(); i++) {
            assertEquals(s1.yTrain.get(i), s2.yTrain.get(i));
        }
    }

    @Test
    void testSmallDataset() {
        Matrix X = new Matrix(new double[][]{{1}});
        Vector y = new Vector(new double[]{0});
        var split = TrainTestSplit.split(X, y, 0.5, 42);
        assertTrue(split.xTrain.rows() > 0 || split.xTest.rows() > 0);
    }
}
