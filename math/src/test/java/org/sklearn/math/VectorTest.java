package org.sklearn.math;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

class VectorTest {

    @Test
    void testCreateAndGet() {
        Vector v = new Vector(new double[]{1.0, 2.0, 3.0});
        assertEquals(3, v.size());
        assertEquals(1.0, v.get(0));
        assertEquals(2.0, v.get(1));
        assertEquals(3.0, v.get(2));
    }

    @Test
    void testZeros() {
        Vector v = Vector.zeros(5);
        for (int i = 0; i < 5; i++) {
            assertEquals(0.0, v.get(i));
        }
    }

    @Test
    void testOnes() {
        Vector v = Vector.ones(4);
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, v.get(i));
        }
    }

    @Test
    void testConstant() {
        Vector v = Vector.constant(3, 7.5);
        for (int i = 0; i < 3; i++) {
            assertEquals(7.5, v.get(i));
        }
    }

    @Test
    void testDot() {
        Vector a = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector b = new Vector(new double[]{4.0, 5.0, 6.0});
        assertEquals(32.0, a.dot(b), 1e-12);
    }

    @Test
    void testNorm() {
        Vector v = new Vector(new double[]{3.0, 4.0});
        assertEquals(5.0, v.norm(), 1e-12);
    }

    @Test
    void testNormSquared() {
        Vector v = new Vector(new double[]{3.0, 4.0});
        assertEquals(25.0, v.normSquared(), 1e-12);
    }

    @Test
    void testSum() {
        Vector v = new Vector(new double[]{1.0, 2.0, 3.0, 4.0});
        assertEquals(10.0, v.sum(), 1e-12);
    }

    @Test
    void testMean() {
        Vector v = new Vector(new double[]{2.0, 4.0, 6.0});
        assertEquals(4.0, v.mean(), 1e-12);
    }

    @Test
    void testMin() {
        Vector v = new Vector(new double[]{3.0, -1.0, 7.0, 0.0});
        assertEquals(-1.0, v.min(), 1e-12);
    }

    @Test
    void testMax() {
        Vector v = new Vector(new double[]{3.0, -1.0, 7.0, 0.0});
        assertEquals(7.0, v.max(), 1e-12);
    }

    @Test
    void testAdd() {
        Vector a = new Vector(new double[]{1.0, 2.0});
        Vector b = new Vector(new double[]{3.0, 4.0});
        Vector c = a.add(b);
        assertEquals(4.0, c.get(0), 1e-12);
        assertEquals(6.0, c.get(1), 1e-12);
    }

    @Test
    void testSubtract() {
        Vector a = new Vector(new double[]{5.0, 7.0});
        Vector b = new Vector(new double[]{3.0, 4.0});
        Vector c = a.subtract(b);
        assertEquals(2.0, c.get(0), 1e-12);
        assertEquals(3.0, c.get(1), 1e-12);
    }

    @Test
    void testMultiplyScalar() {
        Vector v = new Vector(new double[]{1.0, 2.0, 3.0});
        Vector r = v.multiply(2.5);
        assertEquals(2.5, r.get(0), 1e-12);
        assertEquals(5.0, r.get(1), 1e-12);
        assertEquals(7.5, r.get(2), 1e-12);
    }

    @Test
    void testDivideScalar() {
        Vector v = new Vector(new double[]{2.0, 4.0, 6.0});
        Vector r = v.divide(2.0);
        assertEquals(1.0, r.get(0), 1e-12);
        assertEquals(2.0, r.get(1), 1e-12);
        assertEquals(3.0, r.get(2), 1e-12);
    }

    @Test
    void testAbs() {
        Vector v = new Vector(new double[]{-1.0, 2.0, -3.0});
        Vector r = v.abs();
        assertEquals(1.0, r.get(0), 1e-12);
        assertEquals(2.0, r.get(1), 1e-12);
        assertEquals(3.0, r.get(2), 1e-12);
    }

    @Test
    void testSqrt() {
        Vector v = new Vector(new double[]{4.0, 9.0, 16.0});
        Vector r = v.sqrt();
        assertEquals(2.0, r.get(0), 1e-12);
        assertEquals(3.0, r.get(1), 1e-12);
        assertEquals(4.0, r.get(2), 1e-12);
    }

    @Test
    void testAddInPlace() {
        Vector a = new Vector(new double[]{1.0, 2.0});
        Vector b = new Vector(new double[]{3.0, 4.0});
        a.addInPlace(b);
        assertEquals(4.0, a.get(0), 1e-12);
        assertEquals(6.0, a.get(1), 1e-12);
    }

    @Test
    void testMultiplyInPlace() {
        Vector v = new Vector(new double[]{1.0, 2.0, 3.0});
        v.multiplyInPlace(2.0);
        assertEquals(2.0, v.get(0), 1e-12);
        assertEquals(4.0, v.get(1), 1e-12);
        assertEquals(6.0, v.get(2), 1e-12);
    }

    @Test
    void testSlice() {
        Vector v = new Vector(new double[]{0.0, 1.0, 2.0, 3.0, 4.0});
        Vector s = v.slice(1, 4);
        assertEquals(3, s.size());
        assertEquals(1.0, s.get(0), 1e-12);
        assertEquals(2.0, s.get(1), 1e-12);
        assertEquals(3.0, s.get(2), 1e-12);
    }

    @Test
    void testConcatenate() {
        Vector a = new Vector(new double[]{1.0, 2.0});
        Vector b = new Vector(new double[]{3.0, 4.0, 5.0});
        Vector c = Vector.concatenate(a, b);
        assertEquals(5, c.size());
        assertEquals(1.0, c.get(0), 1e-12);
        assertEquals(2.0, c.get(1), 1e-12);
        assertEquals(3.0, c.get(2), 1e-12);
        assertEquals(4.0, c.get(3), 1e-12);
        assertEquals(5.0, c.get(4), 1e-12);
    }

    @Test
    void testToArray() {
        double[] arr = {1.0, 2.0, 3.0};
        Vector v = new Vector(arr);
        double[] result = v.toArray();
        assertArrayEquals(arr, result, 1e-12);
        result[0] = 99.0;
        assertEquals(1.0, v.get(0), 1e-12);
    }

    @Test
    void testDotSizeMismatch() {
        Vector a = new Vector(new double[]{1.0, 2.0});
        Vector b = new Vector(new double[]{1.0, 2.0, 3.0});
        assertThrows(IllegalArgumentException.class, () -> a.dot(b));
    }

    @Test
    void testEmptyVectorMinMax() {
        Vector v = new Vector(0);
        assertThrows(IllegalStateException.class, v::min);
        assertThrows(IllegalStateException.class, v::max);
    }

    @Test
    void testDivideByZero() {
        Vector v = new Vector(new double[]{1.0, 2.0});
        assertThrows(IllegalArgumentException.class, () -> v.divide(0.0));
    }

    @Test
    void testNegativeSize() {
        assertThrows(IllegalArgumentException.class, () -> new Vector(-1));
    }

    @Test
    void testEquality() {
        Vector a = new Vector(new double[]{1.0, 2.0});
        Vector b = new Vector(new double[]{1.0, 2.0});
        Vector c = new Vector(new double[]{1.0, 3.0});
        assertEquals(a, b);
        assertNotEquals(a, c);
    }
}
