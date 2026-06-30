package org.sklearn.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RandomGeneratorTest {

    @Test
    void testDeterministicSeed() {
        RandomGenerator r1 = new RandomGenerator(42);
        RandomGenerator r2 = new RandomGenerator(42);
        for (int i = 0; i < 100; i++) {
            assertEquals(r1.nextDouble(), r2.nextDouble(), 1e-15);
        }
    }

    @Test
    void testNextDoubleRange() {
        RandomGenerator rg = new RandomGenerator(123);
        for (int i = 0; i < 1000; i++) {
            double d = rg.nextDouble();
            assertTrue(d >= 0.0 && d < 1.0);
        }
    }

    @Test
    void testNextIntBound() {
        RandomGenerator rg = new RandomGenerator(456);
        for (int i = 0; i < 1000; i++) {
            int n = rg.nextInt(10);
            assertTrue(n >= 0 && n < 10);
        }
    }

    @Test
    void testFillUniformVector() {
        RandomGenerator rg = new RandomGenerator(789);
        Vector v = new Vector(100);
        rg.fillUniform(v);
        for (int i = 0; i < v.size(); i++) {
            double d = v.get(i);
            assertTrue(d >= 0.0 && d < 1.0);
        }
    }

    @Test
    void testFillNormalMatrix() {
        RandomGenerator rg = new RandomGenerator(101);
        Matrix m = new Matrix(10, 10);
        rg.fillNormal(m);
        double sum = m.sum();
        assertTrue(Double.isFinite(sum));
    }

    @Test
    void testShuffle() {
        RandomGenerator rg = new RandomGenerator(42);
        int[] arr = {0, 1, 2, 3, 4, 5};
        rg.shuffle(arr);
        assertNotEquals(new int[]{0, 1, 2, 3, 4, 5}, arr);
        Arrays.sort(arr);
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5}, arr);
    }

    private static void assertArrayEquals(int[] expected, int[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i]);
        }
    }
}
