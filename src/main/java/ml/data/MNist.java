package ml.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPInputStream;

import math.Matrix;

public final class MNist {
  private MNist() {}

  public static Matrix[] readInput(File inputFile)
    throws IOException
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(inputFile))) {
      byte[] buf = new byte[4];
      fill(in, buf, 4); // magic number

      fill(in, buf, 4); // length
      int inputSize = asInt(buf);

      fill(in, buf, 4);
      int numRows = asInt(buf);

      fill(in, buf, 4);
      int numCols = asInt(buf);

      buf = new byte[inputSize * numRows * numCols];
      fill(in, buf, buf.length);

      Matrix[] input = new Matrix[inputSize];
      int count = 0;
      for (int i = 0; i < inputSize; ++i) {
        input[i] = new Matrix(numRows, numCols);

        for (int j = 0; j < numRows; ++j) {
          for (int k = 0; k < numCols; ++k) {
            input[i].set(j, k, ((double) (buf[count++] & 0xFF)));
          }
        }
      }

      return input;
    }
  }

  public static Matrix readInput(File inputFile, int index)
    throws IOException
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(inputFile))) {
      skip(in, 8);
      byte[] buf = new byte[4];

      fill(in, buf, 4);
      int numRows = asInt(buf);

      fill(in, buf, 4);
      int numCols = asInt(buf);

      skip(in, numRows * numCols * index);

      buf = new byte[numRows * numCols];
      fill(in, buf, buf.length);

      Matrix matrix = new Matrix(numRows, numCols);
      int count = 0;
      for (int j = 0; j < numRows; ++j) {
        for (int k = 0; k < numCols; ++k) {
          matrix.set(j, k, ((double) (buf[count++] & 0xFF)));
        }
      }

      return matrix;
    }
  }

  public static int[] readLabels(File labelFile)
    throws IOException
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(labelFile))) {
      byte[] buf = new byte[4];
      fill(in, buf, 4); // magic number;

      fill(in, buf, 4); // length
      int length = asInt(buf);

      buf = new byte[length];
      fill(in, buf, length);

      int[] labels = new int[length];
      for (int i = 0; i < labels.length; ++i) {
        labels[i] = buf[i] & 0xFF;
      }

      return labels;
    }
  }

  public static BufferedImage matrixToImage(Matrix m) {
    BufferedImage image = new BufferedImage(m.getColumn(), m.getRow(), BufferedImage.TYPE_BYTE_GRAY);
    for (int i = 0; i < m.getRow(); ++i) {
      for (int j = 0; j < m.getColumn(); ++j) {
        image.setRGB(j, i, ((int) m.get(i, j)) * 0x00010101);
      }
    }

    return image;
  }

  public static Matrix imageToMatrix(BufferedImage image) {
    Matrix m = new Matrix(image.getHeight(), image.getWidth());
    for (int i = 0; i < m.getRow(); ++i) {
      for (int j = 0; j < m.getColumn(); ++j) {
        m.set(i, j, (image.getRGB(j, i) & 0xFF));
      }
    }

    return m;
  }

  public static int countImages(File inputFile)
    throws IOException
  {
    try (InputStream in = new GZIPInputStream(new FileInputStream(inputFile))) {
      byte[] buf = new byte[4];
      fill(in, buf, 4); // magic number

      fill(in, buf, 4); // length
      return asInt(buf);
    }
  }

  private static void fill(InputStream in, byte[] buf, int total)
    throws IOException
  {
    int off = 0;
    while (off < total) {
      off += in.read(buf, off, total - off);
    }
  }

  private static int asInt(byte[] bytesInBigEndian) {
    return ByteBuffer.wrap(bytesInBigEndian).order(ByteOrder.BIG_ENDIAN).getInt();
  }

  private static void skip(InputStream in, int skip)
    throws IOException
  {
    int total = 0;
    while (total < skip) {
      total += in.skip(skip - total);
    }
  }
}
