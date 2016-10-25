import static spark.Spark.*;

import javax.servlet.http.HttpServletResponse;

import com.google.gson.Gson;

import math.Matrix;
import ml.nn.DigitRecognition;

public class App {
  public static void main(String[] args) {
    staticFileLocation("/public");

    get("/service/image/count", App::readTestImageCount);
    get("/service/image/:index", App::readTestImage);
    get("/service/image/predict/:index", App::predictTestImage, x -> new Gson().toJson(x));
  }

  private static Object predictTestImage(spark.Request request, spark.Response response) {
    try {
      return DigitRecognition.predict(Integer.parseInt(request.params(":index")));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static Object readTestImageCount(spark.Request request, spark.Response response) {
    response.type("application/json");

    try {
      return DigitRecognition.countTestImages();

    } catch (Exception e) {
      e.printStackTrace();
      return 0;
    }
  }

  private static Object readTestImage(spark.Request request, spark.Response response) {
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
