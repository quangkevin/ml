package ml.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.DoubleFunction;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;

import collection.Tuple;
import math.Matrix;

public class NeuralNetwork {
  private int layers;
  private List<Matrix> bs;
  private List<Matrix> ws;
  private ICostFunction cost = new CrossEntropyCostFunction();
  private DoubleFunction<Double> activationFunction = ActivationFunctions::sigmoid;
  private DoubleFunction<Double> dactivationFunction = ActivationFunctions::dsigmoid;

  public NeuralNetwork(int... sizes) {
    this.layers = sizes.length;
    this.bs = new ArrayList<>(sizes.length - 1);
    this.ws = new ArrayList<>(sizes.length - 1);

    Random random = new Random();

    for (int i = 1; i < sizes.length; ++i) {
      int neurons = sizes[i];
      int incomingNeurons = sizes[i-1];

      bs.add(new Matrix(neurons, 1).set((x, y) -> random.nextGaussian()));
      ws.add(new Matrix(neurons, incomingNeurons).set((x, y) -> random.nextGaussian() / Math.sqrt(incomingNeurons)));
    }
  }

  public NeuralNetwork(File f)
    throws Exception
  {
    restore(f);
  }

  public void train(List<Tuple<Matrix>> training,
                    int epochs,
                    int minBatchSize,
                    double learningRate,
                    double regularizationParameter)
  {
    train(training, epochs, minBatchSize, learningRate, regularizationParameter, null);
  }

  public void train(List<Tuple<Matrix>> training,
                    int epochs,
                    int minBatchSize,
                    double learningRate,
                    double regularizationParameter,
                    Consumer<Integer> consumer)
  {
    minBatchSize = Math.min(minBatchSize, training.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
      System.out.println("epoch: " + epoch);
      Collections.shuffle(training);

      for (int k = 0; k < training.size(); k += minBatchSize) {
        sgd(training.subList(k, Math.min(training.size(), k + minBatchSize)),
            learningRate,
            regularizationParameter,
            training.size());
      }

      if (null != consumer) {
        consumer.accept(epoch);
      }
    }
  }

  public Matrix predict(Matrix x) {
    return feedforward(x);
  }

  public void save(File f)
    throws IOException
  {
    try (DataOutputStream out = new DataOutputStream(new FileOutputStream(f))) {
      out.writeInt(layers);
      for (Matrix b : bs) {
        out.writeInt(b.getRow());
        out.writeInt(b.getColumn());
        for (int i = 0; i < b.getRow(); ++i) {
          for (int j = 0; j < b.getColumn(); ++j) {
            out.writeDouble(b.get(i, j));
          }
        }
      }
      for (Matrix w : ws) {
        out.writeInt(w.getRow());
        out.writeInt(w.getColumn());
        for (int i = 0; i < w.getRow(); ++i) {
          for (int j = 0; j < w.getColumn(); ++j) {
            out.writeDouble(w.get(i, j));
          }
        }
      }
    }
  }

  public void restore(File f)
    throws Exception
  {
    try (DataInputStream in = new DataInputStream(new FileInputStream(f))) {
      this.layers = in.readInt();

      this.bs = new ArrayList<>(this.layers - 1);
      for (int i = 0; i < (this.layers - 1); ++i) {
        Matrix b = new Matrix(in.readInt(), in.readInt());
        for (int j = 0; j < b.getRow(); ++j) {
          for (int k = 0; k < b.getColumn(); ++k) {
            b.set(j, k, in.readDouble());
          }
        }
        bs.add(b);
      }

      this.ws = new ArrayList<>(this.layers - 1);
      for (int i = 0; i < (this.layers - 1); ++i) {
        Matrix w = new Matrix(in.readInt(), in.readInt());
        for (int j = 0; j < w.getRow(); ++j) {
          for (int k = 0; k < w.getColumn(); ++k) {
            w.set(j, k, in.readDouble());
          }
        }

        ws.add(w);
      }
    }
  }

  private void sgd(List<Tuple<Matrix>> training,
                   double learningRate,
                   double regularizationParameter,
                   int totalTrainingSize) {
    List<Matrix> nablaB = zeros(bs);
    List<Matrix> nablaW = zeros(ws);

    for (Tuple<Matrix> xy : training) {
      Tuple<List<Matrix>> deltaNablaBW = backprop(xy);

      zip(nablaB, deltaNablaBW.x, (nb, dnb) -> nb.incrementBy(dnb));
      zip(nablaW, deltaNablaBW.y, (nw, dnw) -> nw.incrementBy(dnw));
    }

    zip(ws, nablaW, (w, nw) -> w
        .multiplyBy(1.0 - learningRate * (regularizationParameter/totalTrainingSize))
        .decrementBy(nw.multiplyBy(learningRate / training.size())));

    zip(bs, nablaB, (b, nb) -> b.decrementBy(nb.multiplyBy(learningRate / training.size())));
  }

  private Matrix feedforward(Matrix x) {
    Matrix a = x;

    for (int i = 0; i < ws.size(); ++i) {
      Matrix w = ws.get(i);
      Matrix b = bs.get(i);

      a = w.dot(a).incrementBy(b).set(activationFunction);
    }

    return a;
  }

  private Tuple<List<Matrix>> backprop(Tuple<Matrix> xy) {
    List<Matrix> nablaB = new ArrayList<>(bs.size());
    List<Matrix> nablaW = new ArrayList<>(ws.size());

    List<Matrix> as = new ArrayList<>(layers);
    as.add(xy.x);

    List<Matrix> zs = new ArrayList<>(layers);

    zip(bs, ws, (b, w) -> {
        zs.add(w.dot(get(as, -1)).incrementBy(b));
        as.add(new Matrix(get(zs, -1)).set(activationFunction));
      });

    Matrix delta = cost.delta(get(zs, -1), get(as, -1), xy.y);
    nablaB.add(delta);
    nablaW.add(delta.dot(get(as, -2).transpose()));

    for (int l = 2; l < layers; ++l) {
      Matrix z = get(zs, -l);
      Matrix sp = new Matrix(z).set(dactivationFunction);

      delta = sp.multiplyBy(get(ws, -l+1).transpose().dot(delta));

      nablaB.add(delta);
      nablaW.add(delta.dot(get(as, -l-1).transpose()));
    }

    Collections.reverse(nablaB);
    Collections.reverse(nablaW);

    return new Tuple<>(nablaB, nablaW);
  }

  private <T> void zip(List<T> a, List<T> b, BiConsumer<T, T> consumer) {
    for (int i = 0; i < a.size(); ++i) {
      consumer.accept(a.get(i), b.get(i));
    }
  }

  private List<Matrix> zeros(List<Matrix> x) {
    List<Matrix> y = new ArrayList<>(x.size());

    for (Matrix i : x) {
      y.add(new Matrix(i.getRow(), i.getColumn()));
    }

    return y;
  }

  private Matrix get(List<Matrix> ms, int index) {
    return (0 <= index
            ? ms.get(index)
            : ms.get(ms.size() + index));

  }

  private Matrix normalize(List<Tuple<Matrix>> trainingSamples) {
    Matrix normalizationFactor = new Matrix(trainingSamples.get(0).x.getRow(), 1);
    for (Tuple<Matrix> m : trainingSamples) {
      normalizationFactor.set((i, j) -> normalizationFactor.get(i,j) + m.x.get(i, j));
    }

    normalizationFactor.set((i, j) -> normalizationFactor.get(i, j) / trainingSamples.size());

    for (Tuple<Matrix> m : trainingSamples) {
      m.x.decrementBy(normalizationFactor);
    }

    return normalizationFactor;
  }
}
