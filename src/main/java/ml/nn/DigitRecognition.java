package ml.nn;

import java.io.File;
import java.util.List;
import java.util.ArrayList;

import collection.Tuple;
import math.Matrix;
import ml.data.MNist;

public class DigitRecognition {
  private static final File DATA_DIR = new File(System.getProperty("datadir"));
  private static final File TRAINING_INPUT_FILE = new File(DATA_DIR, "train-images-idx3-ubyte.gz");
  private static final File TRAINING_LABEL_FILE = new File(DATA_DIR, "train-labels-idx1-ubyte.gz");
  private static final File TEST_INPUT_FILE = new File(DATA_DIR, "t10k-images-idx3-ubyte.gz");
  private static final File TEST_LABEL_FILE = new File(DATA_DIR, "t10k-labels-idx1-ubyte.gz");
  private static final File NETWORK_FILE = new File(DATA_DIR, "network.bin");

  public static void main(String[] arg)
    throws Exception
  {
    //train();
    test();
  }

  public static void train()
    throws Exception
  {
    List<Tuple<Matrix>> training = neuralTuples(MNist.readInput(TRAINING_INPUT_FILE),
                                                MNist.readLabels(TRAINING_LABEL_FILE));

    System.out.println("size: " + training.size());

    NeuralNetwork network = new NeuralNetwork(training.get(0).x.getRow(),
                                              100,
                                              training.get(0).y.getRow());

    network.train(training, 20, 10, .5, 5.0);

    network.save(NETWORK_FILE);
  }

  public static void test()
    throws Exception
  {
    NeuralNetwork network = new NeuralNetwork(NETWORK_FILE);

    List<Tuple<Matrix>> testing = neuralTuples(MNist.readInput(TEST_INPUT_FILE),
                                               MNist.readLabels(TEST_LABEL_FILE));

    int totalGood = 0;

    for (Tuple<Matrix> test : testing) {
      int actual = fromNeuralOutput(test.y);
      int predicted = fromNeuralOutput(network.predict(test.x));
      if (actual == predicted) {
        ++totalGood;
      }
    }

    System.out.printf("Accuracy: " + (((double) totalGood) / testing.size()) * 100.0);
  }

  public static java.awt.image.BufferedImage readTestImage(int index)
    throws Exception
  {
    return MNist.matrixToImage(readTestInput(index));
  }

  private static Matrix readTestInput(int index)
    throws Exception
  {
    return MNist.readInput(TEST_INPUT_FILE, index);
  }

  private static List<Tuple<Matrix>> neuralTuples(Matrix[] x, int[] y) {
    List<Tuple<Matrix>> result = new ArrayList<>(x.length);

    for (int i = 0; i < x.length; ++i) {
      result.add(new Tuple<>(asNeuralInput(x[i]), asNeuralOutput(y[i])));
    }

    return result;
  }

  private static Matrix asNeuralInput(Matrix x) {
    Matrix result = new Matrix(x.getRow() * x.getColumn(), 1);
    for (int i = 0; i < result.getRow(); ++i) {
      result.set(i, 0, x.get(i / x.getColumn(),
                             i % x.getColumn()) / 255.0);
    }

    return result;
  }

  private static Matrix asNeuralOutput(int y) {
    Matrix result = new Matrix(10, 1);
    result.set(y, 0, 1.0);

    return result;
  }

  private static int fromNeuralOutput(Matrix output) {
    int max = 0;
    double maxVal = 0;

    for (int i = 0; i < output.getRow(); ++i) {
      if (maxVal < output.get(i, 0)) {
        max = i;
        maxVal = output.get(i, 0);
      }
    }

    return max;
  }
}
