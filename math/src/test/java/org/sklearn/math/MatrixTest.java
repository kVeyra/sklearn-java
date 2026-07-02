package org.sklearn.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    @Test
    void testCreateAndGet() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(2, m.rows());
        assertEquals(2, m.cols());
        assertEquals(1.0, m.get(0, 0), 1e-12);
        assertEquals(4.0, m.get(1, 1), 1e-12);
    }

    @Test
    void testZeros() {
        Matrix m = Matrix.zeros(3, 4);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(0.0, m.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testOnes() {
        Matrix m = Matrix.ones(2, 3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1.0, m.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testEye() {
        Matrix m = Matrix.eye(3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(i == j ? 1.0 : 0.0, m.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void testFromColumns() {
        Vector a = new Vector(new double[]{1, 4});
        Vector b = new Vector(new double[]{2, 5});
        Vector c = new Vector(new double[]{3, 6});
        Matrix m = Matrix.fromColumns(a, b, c);
        assertEquals(2, m.rows());
        assertEquals(3, m.cols());
        assertEquals(1.0, m.get(0, 0), 1e-12);
        assertEquals(5.0, m.get(1, 1), 1e-12);
    }

    @Test
    void testFromRows() {
        Vector a = new Vector(new double[]{1, 2, 3});
        Vector b = new Vector(new double[]{4, 5, 6});
        Matrix m = Matrix.fromRows(a, b);
        assertEquals(2, m.rows());
        assertEquals(3, m.cols());
        assertEquals(1.0, m.get(0, 0), 1e-12);
        assertEquals(5.0, m.get(1, 1), 1e-12);
    }

    @Test
    void testTranspose() {
        Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix t = m.transpose();
        assertEquals(3, t.rows());
        assertEquals(2, t.cols());
        assertEquals(1.0, t.get(0, 0), 1e-12);
        assertEquals(4.0, t.get(0, 1), 1e-12);
        assertEquals(2.0, t.get(1, 0), 1e-12);
        assertEquals(6.0, t.get(2, 1), 1e-12);
    }

    @Test
    void testMatrixMultiply() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix c = a.multiply(b);
        assertEquals(19.0, c.get(0, 0), 1e-12);
        assertEquals(22.0, c.get(0, 1), 1e-12);
        assertEquals(43.0, c.get(1, 0), 1e-12);
        assertEquals(50.0, c.get(1, 1), 1e-12);
    }

    @Test
    void testMatrixVectorMultiply() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        Vector v = new Vector(new double[]{2, 3});
        Vector r = m.multiply(v);
        assertEquals(3, r.size());
        assertEquals(8.0, r.get(0), 1e-12);
        assertEquals(18.0, r.get(1), 1e-12);
        assertEquals(28.0, r.get(2), 1e-12);
    }

    @Test
    void testMatrixScalarMultiply() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix r = m.multiply(2.0);
        assertEquals(2.0, r.get(0, 0), 1e-12);
        assertEquals(4.0, r.get(0, 1), 1e-12);
        assertEquals(6.0, r.get(1, 0), 1e-12);
        assertEquals(8.0, r.get(1, 1), 1e-12);
    }

    @Test
    void testAdd() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix c = a.add(b);
        assertEquals(6.0, c.get(0, 0), 1e-12);
        assertEquals(8.0, c.get(0, 1), 1e-12);
        assertEquals(10.0, c.get(1, 0), 1e-12);
        assertEquals(12.0, c.get(1, 1), 1e-12);
    }

    @Test
    void testSubtract() {
        Matrix a = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix b = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix c = a.subtract(b);
        assertEquals(4.0, c.get(0, 0), 1e-12);
        assertEquals(4.0, c.get(0, 1), 1e-12);
        assertEquals(4.0, c.get(1, 0), 1e-12);
        assertEquals(4.0, c.get(1, 1), 1e-12);
    }

    @Test
    void testElementMultiply() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix c = a.elementMultiply(b);
        assertEquals(5.0, c.get(0, 0), 1e-12);
        assertEquals(12.0, c.get(0, 1), 1e-12);
        assertEquals(21.0, c.get(1, 0), 1e-12);
        assertEquals(32.0, c.get(1, 1), 1e-12);
    }

    @Test
    void testTrace() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(5.0, m.trace(), 1e-12);
    }

    @Test
    void testSum() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(10.0, m.sum(), 1e-12);
    }

    @Test
    void testMean() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(2.5, m.mean(), 1e-12);
    }

    @Test
    void testInverse() {
        Matrix m = new Matrix(new double[][]{{4, 7}, {2, 6}});
        Matrix inv = m.inverse();
        Matrix identity = m.multiply(inv);
        assertEquals(1.0, identity.get(0, 0), 1e-10);
        assertEquals(0.0, identity.get(0, 1), 1e-10);
        assertEquals(0.0, identity.get(1, 0), 1e-10);
        assertEquals(1.0, identity.get(1, 1), 1e-10);
    }

    @Test
    void testDeterminant() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(-2.0, m.determinant(), 1e-12);
    }

    @Test
    void testDeterminantSingular() {
        Matrix m = new Matrix(new double[][]{{1, 2}, {2, 4}});
        assertEquals(0.0, m.determinant(), 1e-12);
    }

    @Test
    void testRow() {
        Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Vector r = m.row(1);
        assertEquals(3, r.size());
        assertEquals(4.0, r.get(0), 1e-12);
        assertEquals(5.0, r.get(1), 1e-12);
        assertEquals(6.0, r.get(2), 1e-12);
    }

    @Test
    void testCol() {
        Matrix m = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Vector c = m.col(1);
        assertEquals(2, c.size());
        assertEquals(2.0, c.get(0), 1e-12);
        assertEquals(5.0, c.get(1), 1e-12);
    }

    @Test
    void testReshape() {
        Matrix m = new Matrix(new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}});
        Matrix r = m.reshape(4, 2);
        assertEquals(4, r.rows());
        assertEquals(2, r.cols());
        assertEquals(1.0, r.get(0, 0), 1e-12);
        assertEquals(6.0, r.get(2, 1), 1e-12);
    }

    @Test
    void testSolve() {
        Matrix a = new Matrix(new double[][]{{3, 1}, {1, 2}});
        Matrix b = new Matrix(new double[][]{{9}, {8}});
        Matrix x = a.solve(b);
        assertEquals(2.0, x.get(0, 0), 1e-10);
        assertEquals(3.0, x.get(1, 0), 1e-10);
    }

    @Test
    void testDimensionMismatchOnMultiply() {
        Matrix a = new Matrix(new double[][]{{1, 2}});
        Matrix b = new Matrix(new double[][]{{1}, {2}, {3}});
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
    }

    @Test
    void testShapeMismatchOnAdd() {
        Matrix a = new Matrix(2, 2);
        Matrix b = new Matrix(3, 3);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }

    @Test
    void testToArray() {
        double[][] arr = {{1, 2}, {3, 4}};
        Matrix m = new Matrix(arr);
        double[][] result = m.toArray();
        assertArrayEquals(arr[0], result[0], 1e-12);
        assertArrayEquals(arr[1], result[1], 1e-12);
        result[0][0] = 99;
        assertEquals(1.0, m.get(0, 0), 1e-12);
    }
}
