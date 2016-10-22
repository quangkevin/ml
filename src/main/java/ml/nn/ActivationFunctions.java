package ml.nn;

public final class ActivationFunctions {
  private ActivationFunctions() {}

  public static double sigmoid(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  public static double dsigmoid(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }

  public static double tanh(double z) {
    return Math.tanh(z);
  }

  public static double dtanh(double z) {
    double y = Math.tanh(z);

    return (1 - y * y);
  }
}
