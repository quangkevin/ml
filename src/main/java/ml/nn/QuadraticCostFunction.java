package ml.nn;

import java.util.function.DoubleFunction;

import math.Matrix;

public class QuadraticCostFunction implements ICostFunction {
  private DoubleFunction<Double> dactivationFunction;

  public QuadraticCostFunction(DoubleFunction<Double> dactivationFunction) {
    this.dactivationFunction = dactivationFunction;
  }

  public double cost(Matrix a, Matrix y) {
    double total = 0;
    for (int i = 0; i < a.getRow(); ++i) {
      for (int j = 0; j < a.getColumn(); ++i) {
        total += Math.pow(a.get(i, j) - y.get(i, j), 2);
      }
    }

    return total / 2;
  }

  public Matrix delta(Matrix z, Matrix a, Matrix y) {
    return a.subtract(y).set(dactivationFunction);
  }
}
