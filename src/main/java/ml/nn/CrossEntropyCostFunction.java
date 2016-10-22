package ml.nn;

import math.Matrix;

public class CrossEntropyCostFunction implements ICostFunction {
  public double cost(Matrix a, Matrix y) {
    double total = 0;
    for (int i = 0; i < a.getRow(); ++i) {
      for (int j = 0; j < a.getColumn(); ++j) {
        double val = (-y.get(i, j) * Math.log(a.get(i, j))
                      - (1 - y.get(i, j)) * Math.log(1 - a.get(i, j)));
        if (!Double.isNaN(val)) {
          total += val;
        }
      }
    }

    return total;
  }

  public Matrix delta(Matrix z, Matrix a, Matrix y) {
    return a.subtract(y);
  }
}
