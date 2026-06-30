package org.sklearn.math;

import java.util.Arrays;
import java.util.Objects;

public final class Vector {

    private final double[] data;
    private final int size;

    public Vector(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("Size must be non-negative, got: " + size);
        }
        this.size = size;
        this.data = new double[size];
    }

    public Vector(double[] data) {
        Objects.requireNonNull(data, "Data array must not be null");
        this.size = data.length;
        this.data = data.clone();
    }

    public Vector(Vector other) {
        Objects.requireNonNull(other, "Vector must not be null");
        this.size = other.size;
        this.data = other.data.clone();
    }

    public static Vector zeros(int size) {
        return new Vector(size);
    }

    public static Vector ones(int size) {
        Vector v = new Vector(size);
        Arrays.fill(v.data, 1.0);
        return v;
    }

    public static Vector constant(int size, double value) {
        Vector v = new Vector(size);
        Arrays.fill(v.data, value);
        return v;
    }

    public int size() {
        return size;
    }

    public double get(int index) {
        return data[index];
    }

    public void set(int index, double value) {
        data[index] = value;
    }

    public double[] toArray() {
        return data.clone();
    }

    public double dot(Vector other) {
        checkSameSize(other);
        double result = 0.0;
        for (int i = 0; i < size; i++) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    public double norm() {
        return Math.sqrt(dot(this));
    }

    public double normSquared() {
        return dot(this);
    }

    public double sum() {
        double result = 0.0;
        for (int i = 0; i < size; i++) {
            result += data[i];
        }
        return result;
    }

    public double mean() {
        return sum() / size;
    }

    public double min() {
        if (size == 0) {
            throw new IllegalStateException("Cannot compute min of empty vector");
        }
        double m = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] < m) m = data[i];
        }
        return m;
    }

    public double max() {
        if (size == 0) {
            throw new IllegalStateException("Cannot compute max of empty vector");
        }
        double m = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] > m) m = data[i];
        }
        return m;
    }

    public Vector add(Vector other) {
        checkSameSize(other);
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    public Vector subtract(Vector other) {
        checkSameSize(other);
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    public Vector multiply(double scalar) {
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    public Vector divide(double scalar) {
        if (scalar == 0.0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return multiply(1.0 / scalar);
    }

    public Vector abs() {
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = Math.abs(data[i]);
        }
        return result;
    }

    public Vector sqrt() {
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = Math.sqrt(data[i]);
        }
        return result;
    }

    public void addInPlace(Vector other) {
        checkSameSize(other);
        for (int i = 0; i < size; i++) {
            data[i] += other.data[i];
        }
    }

    public void multiplyInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            data[i] *= scalar;
        }
    }

    public Vector slice(int start, int end) {
        if (start < 0 || end > size || start >= end) {
            throw new IllegalArgumentException("Invalid slice bounds");
        }
        double[] sliced = new double[end - start];
        System.arraycopy(data, start, sliced, 0, end - start);
        return new Vector(sliced);
    }

    public static Vector concatenate(Vector... vectors) {
        int totalSize = Arrays.stream(vectors).mapToInt(v -> v.size).sum();
        double[] result = new double[totalSize];
        int offset = 0;
        for (Vector v : vectors) {
            System.arraycopy(v.data, 0, result, offset, v.size);
            offset += v.size;
        }
        return new Vector(result);
    }

    private void checkSameSize(Vector other) {
        if (this.size != other.size) {
            throw new IllegalArgumentException(
                "Vector sizes must match: " + this.size + " vs " + other.size);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Vector vector)) return false;
        return size == vector.size && Arrays.equals(data, vector.data);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "Vector{" + "size=" + size + ", data=" + Arrays.toString(data) + '}';
    }
}
