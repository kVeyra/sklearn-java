package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;

import static org.junit.jupiter.api.Assertions.*;

class MultiLabelBinarizerTest {

    @Test
    void testFitTransform() {
        MultiLabelBinarizer mlb = new MultiLabelBinarizer();
        mlb.fit(new int[][]{{1, 2}, {3}});
        Matrix result = mlb.transform(new int[][]{{1, 3}, {2}});
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(1.0, result.get(0, 2));
        assertEquals(1.0, result.get(1, 1));
    }

    @Test
    void testInverseTransform() {
        MultiLabelBinarizer mlb = new MultiLabelBinarizer();
        mlb.fit(new int[][]{{0, 1}, {2}});
        Matrix binarized = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}});
        int[][] result = mlb.inverseTransform(binarized);
        assertArrayEquals(new int[]{0}, result[0]);
        assertArrayEquals(new int[]{1}, result[1]);
    }

    @Test
    void testGetClasses() {
        MultiLabelBinarizer mlb = new MultiLabelBinarizer();
        mlb.fit(new int[][]{{5}, {3}, {1}});
        assertArrayEquals(new int[]{1, 3, 5}, mlb.getClasses());
    }
}
