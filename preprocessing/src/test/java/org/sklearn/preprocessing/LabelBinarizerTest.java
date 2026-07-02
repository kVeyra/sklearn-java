package org.sklearn.preprocessing;

import org.junit.jupiter.api.Test;
import org.sklearn.math.Matrix;
import org.sklearn.math.Vector;

import static org.junit.jupiter.api.Assertions.*;

class LabelBinarizerTest {

    @Test
    void testFitTransform() {
        LabelBinarizer lb = new LabelBinarizer();
        lb.fit(new Vector(new double[]{1, 2, 3}));
        Matrix result = lb.transform(new Vector(new double[]{1, 3}));
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(1.0, result.get(1, 2));
    }

    @Test
    void testInverseTransform() {
        LabelBinarizer lb = new LabelBinarizer();
        lb.fit(new Vector(new double[]{0, 1, 2}));
        Matrix binarized = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}});
        Vector result = lb.inverseTransform(binarized);
        assertEquals(0.0, result.get(0));
        assertEquals(1.0, result.get(1));
    }

    @Test
    void testGetClasses() {
        LabelBinarizer lb = new LabelBinarizer();
        lb.fit(new Vector(new double[]{5, 3, 1}));
        assertArrayEquals(new int[]{1, 3, 5}, lb.getClasses());
    }
}
