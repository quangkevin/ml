package math;

import java.util.function.BiFunction;
import java.util.function.DoubleFunction;

public class Matrix {
  private double[][] val;

  public Matrix(int row, int col) {
    val = new double[row][col];
  }

  public Matrix(double[][] val) {
    this.val = new double[val.length][0 < val.length ? val[0].length : 0];
    for (int i = 0; i < getRow(); ++i) {
      System.arraycopy(val[i], 0, this.val[i], 0, getColumn());
    }
  }

  public Matrix(Matrix m) {
    this(m.val);
  }

  public Matrix transpose() {
    return new Matrix(getColumn(), getRow())
      .set((i, j) -> get(j, i));
  }

  public Matrix incrementBy(Matrix b) {
    return set((i, j) -> get(i, j) + b.get(i, j));
  }

  public Matrix decrementBy(Matrix b) {
    return set((i, j) -> get(i, j) - b.get(i, j));
  }

  public Matrix multiplyBy(Matrix b) {
    return set((i, j) -> get(i, j) * b.get(i, j));
  }

  public Matrix multiplyBy(double v) {
    return set((i, j) -> get(i, j) * v);
  }

  public Matrix add(Matrix b) {
    Matrix a = this;

    return new Matrix(a.getRow(), a.getColumn())
      .set((i, j) -> a.get(i, j) + b.get(i, j));
  }

  public Matrix subtract(Matrix b) {
    Matrix a = this;

    return new Matrix(a.getRow(), a.getColumn())
      .set((i, j) -> a.get(i, j) - b.get(i, j));
  }

  public Matrix multiply(double v) {
    return new Matrix(getRow(), getColumn())
      .set((i, j) -> get(i, j) * v);
  }

  public Matrix dot(Matrix b) {
    if (getColumn() != b.getRow()) {
      throw new RuntimeException(String.format("Cannot dot %sx%s with %sx%s",
                                               getRow(), getColumn(),
                                               b.getRow(), b.getColumn()));
    }

    Matrix a = this;

    return new Matrix(a.getRow(), b.getColumn())
      .set((i, j) -> {
          double result = 0;
          for (int x = 0; x < this.getColumn(); ++x) {
            result += a.val[i][x] * b.val[x][j];
          }

          return result;
      });
  }

  public double scalar() {
    if (1 != getRow() || 1 != getColumn()) {
      throw new IllegalStateException("Not valid for "
                                      + getRow() + "x" + getColumn()
                                      + " matrix.");
    }

    return get(0, 0);
  }

  public int getRow() { return val.length; }
  public int getColumn() { return 0 < getRow() ? val[0].length : 0; }

  public Matrix set(int i, int j, double value) {
    val[i][j] = value;
    return this;
  }

  public Matrix set(BiFunction<Integer, Integer, Double> f) {
    for (int i = 0; i < val.length; ++i) {
      for (int j = 0; j < val[i].length; ++j) {
        val[i][j] = f.apply(i, j);
      }
    }

    return this;
  }

  public Matrix set(DoubleFunction<Double> f) {
    for (int i = 0; i < val.length; ++i) {
      for (int j = 0; j < val[i].length; ++j) {
        val[i][j] = f.apply(val[i][j]);
      }
    }

    return this;
  }

  public double get(int i, int j) { return val[i][j]; }

  public String toString() {
    StringBuilder buf = new StringBuilder();

    buf.append('[');

    for (int i = 0; i < getRow(); ++i) {
      if (0 < i) {
        buf.append(",\n ");
      }

      buf.append('[');
      for (int j = 0; j < getColumn(); ++j) {
        if (0 < j) buf.append(", ");
        //buf.append(String.format("%5.3f", val[i][j]));
        buf.append(val[i][j]);
      }
      buf.append("]");
    }

    buf.append("]");

    return buf.toString();
  }
}
