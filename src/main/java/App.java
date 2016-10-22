import static spark.Spark.*;

import javax.servlet.http.HttpServletResponse;

import math.Matrix;
import ml.nn.DigitRecognition;

public class App {
  public static void main(String[] args) {
    staticFileLocation("/public");

    get("/mnist/test/:index", App::readMNistTestImage);
  }

  private static Object readMNistTestImage(spark.Request request, spark.Response response) {
    int index = Integer.parseInt(request.params(":index"));

    HttpServletResponse raw = response.raw();
    response.type("application/jpg");

    try {
      javax.imageio.ImageIO.write(DigitRecognition.readTestImage(index),
                                  "jpg",
                                  raw.getOutputStream());

      raw.getOutputStream().flush();
      raw.getOutputStream().close();

    } catch (Exception e) {
      e.printStackTrace();
    }

    return raw;
  }
}
