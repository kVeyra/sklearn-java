package org.sklearn.math;

import java.util.Arrays;
import java.util.Objects;

/**
 * A dense matrix of double values stored in row-major order.
 *
 * <p>Provides matrix arithmetic (add, subtract, multiply, transpose),
 * linear algebra operations (inverse via LU decomposition, determinant,
 * solve), and utility methods (reshape, trace, sum, mean).
 *
 * <p>All operations produce new matrices unless documented as in-place.
 */
public final class Matrix {

    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException(
                "Dimensions must be non-negative, got: " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        Objects.requireNonNull(data, "Data must not be null");
        this.rows = data.length;
        this.cols = (rows == 0) ? 0 : data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException(
                    "Row " + i + " has length " + data[i].length + ", expected " + cols);
            }
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    public Matrix(Matrix other) {
        Objects.requireNonNull(other, "Matrix must not be null");
        this.rows = other.rows;
        this.cols = other.cols;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(other.data[i], 0, this.data[i], 0, cols);
        }
    }

    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    public static Matrix ones(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            Arrays.fill(m.data[i], 1.0);
        }
        return m;
    }

    public static Matrix eye(int n) {
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            m.data[i][i] = 1.0;
        }
        return m;
    }

    public static Matrix constant(int rows, int cols, double value) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            Arrays.fill(m.data[i], value);
        }
        return m;
    }

    public static Matrix fromColumns(Vector... columns) {
        int cols = columns.length;
        int rows = cols == 0 ? 0 : columns[0].size();
        Matrix m = new Matrix(rows, cols);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                m.data[i][j] = columns[j].get(i);
            }
        }
        return m;
    }

    public static Matrix fromRows(Vector... rows) {
        int rowsCount = rows.length;
        int cols = rowsCount == 0 ? 0 : rows[0].size();
        Matrix m = new Matrix(rowsCount, cols);
        for (int i = 0; i < rowsCount; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = rows[i].get(j);
            }
        }
        return m;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int[] shape() {
        return new int[]{rows, cols};
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    public double[][] toArray() {
        double[][] copy = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    public Vector row(int index) {
        return new Vector(data[index]);
    }

    public Vector col(int index) {
        double[] colData = new double[rows];
        for (int i = 0; i < rows; i++) {
            colData[i] = data[i][index];
        }
        return new Vector(colData);
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "Matrix dimensions mismatch: " + this.rows + "x" + this.cols
                    + " vs " + other.rows + "x" + other.cols);
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int k = 0; k < this.cols; k++) {
                double aik = data[i][k];
                if (aik != 0.0) {
                    for (int j = 0; j < other.cols; j++) {
                        result.data[i][j] += aik * other.data[k][j];
                    }
                }
            }
        }
        return result;
    }

    public Vector multiply(Vector vector) {
        if (vector.size() != cols) {
            throw new IllegalArgumentException(
                "Vector size must match matrix columns: " + vector.size() + " vs " + cols);
        }
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += data[i][j] * vector.get(j);
            }
            result[i] = sum;
        }
        return new Vector(result);
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    public Matrix add(Matrix other) {
        checkSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        checkSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    public Matrix elementMultiply(Matrix other) {
        checkSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public double trace() {
        if (rows != cols) {
            throw new IllegalStateException("Trace requires square matrix");
        }
        double t = 0.0;
        for (int i = 0; i < rows; i++) {
            t += data[i][i];
        }
        return t;
    }

    public double sum() {
        double s = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                s += data[i][j];
            }
        }
        return s;
    }

    public double mean() {
        return sum() / (rows * cols);
    }

    public Matrix reshape(int newRows, int newCols) {
        if (rows * cols != newRows * newCols) {
            throw new IllegalArgumentException(
                "Cannot reshape " + rows + "x" + cols + " to " + newRows + "x" + newCols);
        }
        Matrix result = new Matrix(newRows, newCols);
        int r = 0, c = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[r][c] = data[i][j];
                c++;
                if (c == newCols) {
                    c = 0;
                    r++;
                }
            }
        }
        return result;
    }

    public Matrix solve(Matrix b) {
        if (rows != cols) {
            throw new IllegalStateException("Solve requires square matrix");
        }
        return solveViaLU(b);
    }

    private Matrix solveViaLU(Matrix b) {
        int n = rows;
        double[][] a = toArray();
        double[][] bArr = b.toArray();
        int bCols = b.cols;

        int[] pivot = new int[n];
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }

        for (int col = 0; col < n - 1; col++) {
            double maxVal = Math.abs(a[col][col]);
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                double val = Math.abs(a[row][col]);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = row;
                }
            }
            if (maxVal < 1e-15) {
                throw new IllegalStateException("Singular or nearly singular matrix");
            }
            if (maxRow != col) {
                double[] tmp = a[col]; a[col] = a[maxRow]; a[maxRow] = tmp;
                int tmpP = pivot[col]; pivot[col] = pivot[maxRow]; pivot[maxRow] = tmpP;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = a[row][col] / a[col][col];
                a[row][col] = factor;
                for (int j = col + 1; j < n; j++) {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }

        Matrix result = new Matrix(n, bCols);
        for (int j = 0; j < bCols; j++) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = bArr[pivot[i]][j];
                for (int k = 0; k < i; k++) {
                    x[i] -= a[i][k] * x[k];
                }
            }
            for (int i = n - 1; i >= 0; i--) {
                for (int k = i + 1; k < n; k++) {
                    x[i] -= a[i][k] * x[k];
                }
                x[i] /= a[i][i];
            }
            for (int i = 0; i < n; i++) {
                result.data[i][j] = x[i];
            }
        }
        return result;
    }

    public Matrix inverse() {
        return solve(eye(rows));
    }

    public double determinant() {
        if (rows != cols) {
            throw new IllegalStateException("Determinant requires square matrix");
        }
        int n = rows;
        double[][] a = toArray();
        double det = 1.0;
        for (int col = 0; col < n - 1; col++) {
            double maxVal = Math.abs(a[col][col]);
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                double val = Math.abs(a[row][col]);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = row;
                }
            }
            if (maxVal < 1e-15) {
                return 0.0;
            }
            if (maxRow != col) {
                double[] tmp = a[col]; a[col] = a[maxRow]; a[maxRow] = tmp;
                det = -det;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = a[row][col] / a[col][col];
                for (int j = col + 1; j < n; j++) {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            det *= a[i][i];
        }
        return det;
    }

    private void checkSameShape(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(
                "Matrix shapes must match: " + this.rows + "x" + this.cols
                    + " vs " + other.rows + "x" + other.cols);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Matrix matrix)) {
            return false;
        }
        if (rows != matrix.rows || cols != matrix.cols) {
            return false;
        }
        return Arrays.deepEquals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        return Arrays.deepHashCode(data);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix{").append(rows).append("x").append(cols).append("\n");
        for (int i = 0; i < rows && i < 10; i++) {
            sb.append("  ").append(Arrays.toString(data[i])).append("\n");
        }
        if (rows > 10) {
            sb.append("  ...\n");
        }
        sb.append("}");
        return sb.toString();
    }
}
