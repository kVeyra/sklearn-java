package org.sklearn.math;

import java.util.Random;

/**
 * A deterministic random number generator for sklearn-java.
 *
 * <p>Wraps {@link java.util.Random} with a fixed seed for reproducibility.
 * Mirrors sklearn's {@code random_state} behavior.
 */
public final class RandomGenerator {

    private final Random random;

    /**
     * Create a generator with the given seed for deterministic behavior.
     */
    public RandomGenerator(long seed) {
        this.random = new Random(seed);
    }

    /**
     * Create a generator with a system-nondeterministic seed.
     * Results will NOT be reproducible.
     */
    public RandomGenerator() {
        this.random = new Random();
    }

    /**
     * Return the next pseudo-random, uniformly distributed double
     * in the range [0.0, 1.0).
     */
    public double nextDouble() {
        return random.nextDouble();
    }

    /**
     * Return the next pseudo-random, uniformly distributed double
     * in the range [0.0, max).
     */
    public double nextDouble(double max) {
        return random.nextDouble() * max;
    }

    /**
     * Return the next pseudo-random, uniformly distributed int
     * in the range [0, bound).
     */
    public int nextInt(int bound) {
        return random.nextInt(bound);
    }

    /**
     * Return the next pseudo-random Gaussian (normal) double
     * with mean 0.0 and standard deviation 1.0.
     */
    public double nextGaussian() {
        return random.nextGaussian();
    }

    /**
     * Fill the given vector with uniform random values in [0.0, 1.0).
     */
    public void fillUniform(Vector v) {
        for (int i = 0; i < v.size(); i++) {
            v.set(i, random.nextDouble());
        }
    }

    /**
     * Fill the given vector with standard normal values.
     */
    public void fillNormal(Vector v) {
        for (int i = 0; i < v.size(); i++) {
            v.set(i, random.nextGaussian());
        }
    }

    /**
     * Fill the given matrix with uniform random values in [0.0, 1.0).
     */
    public void fillUniform(Matrix m) {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                m.set(i, j, random.nextDouble());
            }
        }
    }

    /**
     * Fill the given matrix with standard normal values.
     */
    public void fillNormal(Matrix m) {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                m.set(i, j, random.nextGaussian());
            }
        }
    }

    /**
     * Shuffle the given array in-place.
     */
    public void shuffle(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }
}
