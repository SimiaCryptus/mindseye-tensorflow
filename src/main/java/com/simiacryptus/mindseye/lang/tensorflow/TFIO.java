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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefLongStream;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.DataType;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.stream.Stream;

public @RefAware
class TFIO {

  public static TensorArray getTensorList(org.tensorflow.Tensor<?> tensor) {
    return getTensorList(tensor, true);
  }

  public static TensorArray getTensorList(org.tensorflow.Tensor<?> tensor, boolean invertRanks) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensorArray_Double(tensor.expect(Double.class), tensor.shape(), invertRanks);
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensorArray_Float(tensor.expect(Float.class), tensor.shape(), invertRanks);
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  public static Tensor getTensor(org.tensorflow.Tensor<?> tensor) {
    return getTensor(tensor, true);
  }

  public static Tensor getTensor(org.tensorflow.Tensor<?> tensor, boolean invertRanks) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensor_Double(tensor.expect(Double.class), tensor.shape(), invertRanks);
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensor_Float(tensor.expect(Float.class), tensor.shape(), invertRanks);
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(Tensor data) {
    return getFloatTensor(data, true);
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(Tensor data, boolean invertRanks) {
    Tensor invertDimensions;
    double[] buffer;
    if (invertRanks) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    return org.tensorflow.Tensor.create(Util.toLong(data.getDimensions()), FloatBuffer.wrap(Util.getFloats(buffer)));
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(TensorList data) {
    return getFloatTensor(data, true);
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(TensorList data, boolean invertRanks) {
    double[] buffer = getDoubles(data, invertRanks);
    long[] shape = RefLongStream
        .concat(RefLongStream.of(data.length()),
            RefArrays.stream(data.getDimensions()).mapToLong(x -> x))
        .toArray();
    @NotNull
    org.tensorflow.@NotNull Tensor<Float> tensor = org.tensorflow.Tensor.create(shape,
        FloatBuffer.wrap(Util.getFloats(buffer)));
    RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    return tensor;
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(Tensor data) {
    return getDoubleTensor(data, true);
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(Tensor data, boolean invertRanks) {
    double[] buffer;
    Tensor invertDimensions;
    if (invertRanks) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    return org.tensorflow.Tensor.create(Util.toLong(data.getDimensions()), DoubleBuffer.wrap(buffer));
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(TensorList data) {
    return getDoubleTensor(data, true);
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(TensorList data, boolean invertRanks) {
    double[] buffer = getDoubles(data, invertRanks);
    long[] shape = RefLongStream
        .concat(RefLongStream.of(data.length()),
            RefArrays.stream(data.getDimensions()).mapToLong(x -> x))
        .toArray();
    org.tensorflow.Tensor<Double> tensor = org.tensorflow.Tensor.create(shape, DoubleBuffer.wrap(buffer));
    RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    return tensor;
  }

  private static void free(Object obj) {
    if (obj instanceof double[]) {
      double[] doubles = (double[]) obj;
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
    } else if (obj instanceof float[]) {
      float[] floats = (float[]) obj;
      RecycleBin.FLOATS.recycle(floats, floats.length);
    } else {
      RefArrays.stream((Object[]) obj).forEach(x -> free(x));
    }
  }

  private static Object createFloatArray(long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.FLOATS.obtain(shape[0]);
    } else if (shape.length == 2) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> new float[(int) shape[1]]).toArray(s -> new float[s][]);
    } else if (shape.length == 3) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> new float[(int) shape[2]]).toArray(s -> new float[s][]))
          .toArray(s -> new float[s][][]);
    } else if (shape.length == 4) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> new float[(int) shape[3]]).toArray(s -> new float[s][]))
              .toArray(s -> new float[s][][]))
          .toArray(s -> new float[s][][][]);
    } else if (shape.length == 5) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> RefIntStream.range(0, (int) shape[3])
                      .mapToObj(l -> new float[(int) shape[4]]).toArray(s -> new float[s][]))
                  .toArray(s -> new float[s][][]))
              .toArray(s -> new float[s][][][]))
          .toArray(s -> new float[s][][][][]);
    } else if (shape.length == 6) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> RefIntStream.range(0, (int) shape[3])
                      .mapToObj(l -> RefIntStream.range(0, (int) shape[4])
                          .mapToObj(m -> new float[(int) shape[5]]).toArray(s -> new float[s][]))
                      .toArray(s -> new float[s][][]))
                  .toArray(s -> new float[s][][][]))
              .toArray(s -> new float[s][][][][]))
          .toArray(s -> new float[s][][][][][]);
    } else {
      throw new RuntimeException("Rank " + shape.length);
    }
  }

  private static Object createDoubleArray(long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.DOUBLES.obtain(shape[0]);
    } else if (shape.length == 2) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> new double[(int) shape[1]]).toArray(s -> new double[s][]);
    } else if (shape.length == 3) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> new double[(int) shape[2]]).toArray(s -> new double[s][]))
          .toArray(s -> new double[s][][]);
    } else if (shape.length == 4) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> new double[(int) shape[3]]).toArray(s -> new double[s][]))
              .toArray(s -> new double[s][][]))
          .toArray(s -> new double[s][][][]);
    } else {
      throw new RuntimeException("Rank " + shape.length);
    }
  }

  private static RefDoubleStream flattenDoubles(Object obj) {
    if (obj instanceof double[]) {
      return RefArrays.stream((double[]) obj);
    } else if (obj instanceof Double) {
      return RefDoubleStream.of((double) obj);
    } else {
      return RefArrays.stream((Object[]) obj).flatMapToDouble(TFIO::flattenDoubles);
    }
  }

  private static Stream<Float> flattenFloats(Object floats) {
    if (floats instanceof float[]) {
      float[] array = (float[]) floats;
      return Floats.asList(array).stream();
    } else {
      return RefArrays.stream((Object[]) floats).flatMap(x -> flattenFloats(x));
    }
  }

  private static double[] getDoubles(TensorList data, boolean invertRanks) {
    double[] buffer = RecycleBin.DOUBLES.obtain(data.length() * Tensor.length(data.getDimensions()));
    DoubleBuffer inputBuffer = DoubleBuffer.wrap(buffer);
    if (invertRanks) {
      data.stream().map(x -> x.invertDimensions()).forEach(t -> {
        inputBuffer.put(t.getData());
      });
    } else {
      data.stream().forEach(t -> {
        inputBuffer.put(t.getData());
      });
    }
    return buffer;
  }

  private static TensorArray getTensorArray_Float(org.tensorflow.Tensor<Float> tensor, long[] shape,
                                                  boolean invertRanks) {
    float[] doubles = getFloats(tensor);
    int[] dims = RefArrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = new TensorArray(
        RefIntStream.range(0, batches).mapToObj(i -> {
          int offset = i * Tensor.length(dims);
          if (invertRanks) {
            Tensor returnValue = new Tensor(Tensor.reverse(dims));
            for (int j = 0; j < returnValue.length(); j++) {
              returnValue.getData()[j] = doubles[j + offset];
            }
            return returnValue.invertDimensions();
          } else {
            Tensor returnValue = new Tensor(dims);
            for (int j = 0; j < returnValue.length(); j++) {
              returnValue.getData()[j] = doubles[j + offset];
            }
            return returnValue;
          }
        }).toArray(i -> new Tensor[i]));
    RecycleBin.FLOATS.recycle(doubles, doubles.length);
    return resultData;
  }

  private static Tensor getTensor_Float(org.tensorflow.Tensor<Float> tensor, long[] shape, boolean invertRanks) {
    if (0 == tensor.numElements())
      return new Tensor(RefArrays.stream(shape).mapToInt(x -> (int) x).toArray());
    float[] doubles = getFloats(tensor);
    int[] dims = RefArrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if (invertRanks) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      for (int j = 0; j < returnValue.length(); j++) {
        returnValue.getData()[j] = doubles[j];
      }
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return returnValue.invertDimensions();
    } else {
      Tensor returnValue = new Tensor(dims);
      for (int j = 0; j < returnValue.length(); j++) {
        returnValue.getData()[j] = doubles[j];
      }
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return returnValue;
    }
  }

  private static TensorArray getTensorArray_Double(org.tensorflow.Tensor<Double> tensor, long[] shape,
                                                   boolean invertRanks) {
    double[] doubles = getDoubles(tensor);
    int[] dims = RefArrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = new TensorArray(
        RefIntStream.range(0, batches).mapToObj(i -> {
          if (invertRanks) {
            Tensor returnValue = new Tensor(Tensor.reverse(dims));
            System.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0, returnValue.length());
            return returnValue.invertDimensions();
          } else {
            Tensor returnValue = new Tensor(dims);
            System.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0, returnValue.length());
            return returnValue;
          }
        }).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(doubles, doubles.length);
    return resultData;
  }

  private static Tensor getTensor_Double(org.tensorflow.Tensor<Double> tensor, long[] shape, boolean invertRanks) {
    double[] doubles = getDoubles(tensor);
    int[] dims = RefArrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if (invertRanks) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      System.arraycopy(doubles, 0, returnValue.getData(), 0, returnValue.length());
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      return returnValue.invertDimensions();
    } else {
      Tensor returnValue = new Tensor(dims);
      System.arraycopy(doubles, 0, returnValue.getData(), 0, returnValue.length());
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      return returnValue;
    }
  }

  private static double[] getDoubles(org.tensorflow.Tensor<Double> result) {
    Object deepArray = result.copyTo(createDoubleArray(result.shape()));
    double[] doubles = flattenDoubles(deepArray).toArray();
    free(deepArray);
    return doubles;
  }

  private static float[] getFloats(org.tensorflow.Tensor<Float> result) {
    if (0 == result.numElements())
      return new float[]{};
    Object deepArray = result.copyTo(createFloatArray(result.shape()));
    double[] doubles = flattenFloats(deepArray).mapToDouble(x -> x).toArray();
    free(deepArray);
    return Util.getFloats(doubles);
  }

}
