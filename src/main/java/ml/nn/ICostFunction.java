package ml.nn;

import math.Matrix;

public interface ICostFunction {
  public Matrix delta(Matrix z, Matrix a, Matrix y);
  public double cost(Matrix a, Matrix y);
}
