/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.lang.tensorflow;

import com.google.common.primitives.Floats;
import com.simiacryptus.lang.UncheckedConsumer;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TFUtil {

  public static double[] doublesToDoubles(org.tensorflow.Tensor<Double> result) {
    Object deepArray = result.copyTo(createDoubleArray(result.shape()));
    double[] doubles = flattenDoubles(deepArray).toArray();
    free(deepArray);
    return doubles;
  }

  public static float[] floatsToDoubles(org.tensorflow.Tensor<Float> result) {
    Object deepArray = result.copyTo(createFloatArray(result.shape()));
    double[] doubles = flattenFloats(deepArray).mapToDouble(x->x).toArray();
    free(deepArray);
    return getFloats(doubles);
  }

  private static void free(Object obj) {
    if (obj instanceof double[]) {
      double[] doubles = (double[]) obj;
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
    } else if (obj instanceof float[]) {
      float[] floats = (float[]) obj;
      RecycleBin.FLOATS.recycle(floats, floats.length);
    } else {
      Arrays.stream((Object[]) obj).forEach(x -> free(x));
    }

  }

  public static Object createFloatArray(long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.FLOATS.obtain(shape[0]);
    } else if (shape.length == 2) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> new float[(int) shape[1]])
          .toArray(s -> new float[s][]);
    } else if (shape.length == 3) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> IntStream.range(0, (int) shape[1])
              .mapToObj(j -> new float[(int) shape[2]])
              .toArray(s -> new float[s][]))
          .toArray(s -> new float[s][][]);
    } else if (shape.length == 4) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> IntStream.range(0, (int) shape[1])
              .mapToObj(j -> IntStream.range(0, (int) shape[2])
                  .mapToObj(k -> new float[(int) shape[3]])
                  .toArray(s -> new float[s][]))
              .toArray(s -> new float[s][][]))
          .toArray(s -> new float[s][][][]);
    } else {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> createFloatArray(Arrays.stream(shape).skip(1).toArray()))
          .toArray(s -> new Object[s]);
    }
  }
  public static Object createDoubleArray(long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.DOUBLES.obtain(shape[0]);
    } else if (shape.length == 2) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> new double[(int) shape[1]])
          .toArray(s -> new double[s][]);
    } else if (shape.length == 3) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> IntStream.range(0, (int) shape[1])
              .mapToObj(j -> new double[(int) shape[2]])
              .toArray(s -> new double[s][]))
          .toArray(s -> new double[s][][]);
    } else if (shape.length == 4) {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> IntStream.range(0, (int) shape[1])
              .mapToObj(j -> IntStream.range(0, (int) shape[2])
                  .mapToObj(k -> new double[(int) shape[3]])
                  .toArray(s -> new double[s][]))
              .toArray(s -> new double[s][][]))
          .toArray(s -> new double[s][][][]);
    } else {
      return IntStream.range(0, (int) shape[0])
          .mapToObj(i -> createDoubleArray(Arrays.stream(shape).skip(1).toArray()))
          .toArray(s -> new Object[s]);
    }
  }

  public static DoubleStream flattenDoubles(Object floats) {
    if (floats instanceof double[]) {
      return Arrays.stream((double[]) floats);
    } else {
      return Arrays.stream((Object[]) floats).flatMapToDouble(x -> flattenDoubles(x));
    }
  }
  public static Stream<Float> flattenFloats(Object floats) {
    if (floats instanceof float[]) {
      float[] array = (float[]) floats;
      return Floats.asList(array).stream();
    } else {
      return Arrays.stream((Object[]) floats).flatMap(x -> flattenFloats(x));
    }
  }

  @NotNull
  public static TensorArray toTensorArray(long[] shape, float[] entireBuffer) {
    int[] dims = Arrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    return toTensorArray(entireBuffer, dims, batches);
  }

  @NotNull
  public static TensorArray toTensorArray(long[] shape, double[] entireBuffer) {
    int rank = shape.length - 1;
    int[] dims = Arrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    return toTensorArray(entireBuffer, dims, batches);
  }

  @NotNull
  public static TensorArray toTensorArray(double[] entireBuffer, int[] dims) {
    int batches = entireBuffer.length / Arrays.stream(dims).reduce((a, b) -> a * b).getAsInt();
    return toTensorArray(entireBuffer, dims, batches);
  }

  @NotNull
  public static TensorArray toTensorArray(double[] entireBuffer, int[] dims, int batches) {
    return TensorArray.wrap(IntStream.range(0, batches).mapToObj(i -> {
      com.simiacryptus.mindseye.lang.Tensor tensor = new com.simiacryptus.mindseye.lang.Tensor(dims);
      System.arraycopy(entireBuffer, i * tensor.length(), tensor.getData(), 0, tensor.length());
      return tensor;
    }).toArray(i -> new com.simiacryptus.mindseye.lang.Tensor[i]));
  }
  @NotNull
  public static TensorArray toTensorArray(float[] entireBuffer, int[] dims, int batches) {
    return TensorArray.wrap(IntStream.range(0, batches).mapToObj(i -> {
      Tensor tensor = new Tensor(dims);
      for (int j = 0; j < tensor.length(); j++) {
        tensor.getData()[j] = entireBuffer[j + i*tensor.length()];
      }
      return tensor;
    }).toArray(i -> new com.simiacryptus.mindseye.lang.Tensor[i]));
  }

  public static void launchTensorboard(File logDir, UncheckedConsumer<Process> waiter) throws IOException, URISyntaxException {
    Process tensorboard = new ProcessBuilder().command(
        System.getProperty("tensorboard", System.getProperty("user.home") + "\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\tensorboard.exe"),
        "--logdir=" + logDir.getAbsolutePath()
    ).start();
    Desktop.getDesktop().browse(new URI("http://localhost:6006/"));
    try {
      try {
        waiter.accept(tensorboard);
      } catch (RuntimeException e) {
        throw e;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    } finally {
      if (tensorboard.isAlive()) tensorboard.destroyForcibly();
    }
  }

  public static double[] getDoubles(TensorList data) {
    double[] buffer = RecycleBin.DOUBLES.obtain(data.length() * Tensor.length(data.getDimensions()));
    DoubleBuffer inputBuffer = DoubleBuffer.wrap(buffer);
    data.stream().forEach(t -> {
      inputBuffer.put(t.getData());
      t.freeRef();
    });
    return buffer;
  }

  @NotNull
  public static GraphDef implantConstants(GraphDef graphDef, Map<String, Tensor> weights) {
    graphDef = TensorflowUtil.editGraph(graphDef, graphBuilder -> {
      weights.forEach((key, value) -> {
        TensorflowUtil.editNode(graphBuilder, key, (NodeDef.Builder node) -> {
              DataType type = node.getAttrMap().get("dtype").getType();
              TensorProto.Builder tensor = TensorProto.newBuilder();
              if (type == DataType.DT_DOUBLE) {
                tensor.addAllDoubleVal(Arrays.stream(value.getData()).mapToObj(x -> x).collect(Collectors.toList()));
              } else if (type == DataType.DT_FLOAT) {
                tensor.addAllFloatVal(Arrays.stream(value.getData()).mapToObj(x -> (float)x).collect(Collectors.toList()));
              } else {
                throw new UnsupportedOperationException(type.toString());
              }
              tensor.setDtype(type);
              tensor.setTensorShape(node.getAttrMap().get("shape").getShape());
              return node
                  .removeAttr("shape")
                  .putAttr("value", AttrValue.newBuilder()
                      .setTensor(tensor.build())
                      .build())
                  .setOp("Const");
            }
        );
      });
      return graphBuilder;
    });
    return graphDef;
  }

  public static float[] getFloats(double[] buffer) {
    float[] floats = new float[buffer.length];
    for (int i = 0; i < buffer.length; i++) {
      floats[i] = (float) buffer[i];
    }
    return floats;
  }

  public static double[] toDoubles(float[] floats) {
    double[] doubles = new double[floats.length];
    for (int i = 0; i < floats.length; i++) {
      doubles[i] = floats[i];
    }
    return doubles;
  }
}
