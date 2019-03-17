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
import com.simiacryptus.lang.ref.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.DataType;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.*;

public class TFIO {

  private static final boolean INVERT_RANKS = true;

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

  private static Object createFloatArray(long[] shape) {
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

  private static Object createDoubleArray(long[] shape) {
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

  private static DoubleStream flattenDoubles(Object obj) {
    if (obj instanceof double[]) {
      return Arrays.stream((double[]) obj);
    } else if (obj instanceof Double) {
      return DoubleStream.of((double) obj);
    } else {
      return Arrays.stream((Object[]) obj).flatMapToDouble(TFIO::flattenDoubles);
    }
  }

  private static Stream<Float> flattenFloats(Object floats) {
    if (floats instanceof float[]) {
      float[] array = (float[]) floats;
      return Floats.asList(array).stream();
    } else {
      return Arrays.stream((Object[]) floats).flatMap(x -> flattenFloats(x));
    }
  }

  private static double[] getDoubles(TensorList data) {
    double[] buffer = RecycleBin.DOUBLES.obtain(data.length() * Tensor.length(data.getDimensions()));
    DoubleBuffer inputBuffer = DoubleBuffer.wrap(buffer);
    if(INVERT_RANKS) {
      data.stream().map(Tensor::invertDimensionsAndFree).forEach(t -> {
        inputBuffer.put(t.getData());
        t.freeRef();
      });
    } else {
      data.stream().forEach(t -> {
        inputBuffer.put(t.getData());
        t.freeRef();
      });
    }
    return buffer;
  }

  private static TensorArray getTensorArray_Float(org.tensorflow.Tensor<Float> tensor, long[] shape) {
    float[] doubles = getFloats(tensor);
    int[] dims = Arrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = TensorArray.wrap(IntStream.range(0, batches).mapToObj(i -> {
      if(INVERT_RANKS) {
        Tensor returnValue = new Tensor(Tensor.reverse(dims));
        for (int j = 0; j < returnValue.length(); j++) {
          returnValue.getData()[j] = doubles[j + i * returnValue.length()];
        }
        return returnValue.invertDimensionsAndFree();
      } else {
        Tensor returnValue = new Tensor(dims);
        for (int j = 0; j < returnValue.length(); j++) {
          returnValue.getData()[j] = doubles[j + i * returnValue.length()];
        }
        return returnValue;
      }
    }).toArray(i -> new Tensor[i]));
    RecycleBin.FLOATS.recycle(doubles, doubles.length);
    return resultData;
  }

  private static Tensor getTensor_Float(org.tensorflow.Tensor<Float> tensor, long[] shape) {
    float[] doubles = getFloats(tensor);
    int[] dims = Arrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if(INVERT_RANKS) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      for (int j = 0; j < returnValue.length(); j++) {
        returnValue.getData()[j] = doubles[j];
      }
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return returnValue.invertDimensionsAndFree();
    } else {
      Tensor returnValue = new Tensor(dims);
      for (int j = 0; j < returnValue.length(); j++) {
        returnValue.getData()[j] = doubles[j];
      }
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return returnValue;
    }
  }

  private static TensorArray getTensorArray_Double(org.tensorflow.Tensor<Double> tensor, long[] shape) {
    double[] doubles = getDoubles(tensor);
    int[] dims = Arrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = TensorArray.wrap(IntStream.range(0, batches).mapToObj(i -> {
      if(INVERT_RANKS) {
        Tensor returnValue = new Tensor(Tensor.reverse(dims));
        System.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0, returnValue.length());
        return returnValue.invertDimensionsAndFree();
      } else {
        Tensor returnValue = new Tensor(dims);
        System.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0, returnValue.length());
        return returnValue;
      }
    }).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(doubles, doubles.length);
    return resultData;
  }

  private static Tensor getTensor_Double(org.tensorflow.Tensor<Double> tensor, long[] shape) {
    double[] doubles = getDoubles(tensor);
    int[] dims = Arrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if(INVERT_RANKS) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      System.arraycopy(doubles, 0, returnValue.getData(), 0, returnValue.length());
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      return returnValue.invertDimensionsAndFree();
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
    Object deepArray = result.copyTo(createFloatArray(result.shape()));
    double[] doubles = flattenFloats(deepArray).mapToDouble(x -> x).toArray();
    free(deepArray);
    return Util.getFloats(doubles);
  }

  public static TensorArray getTensorList(org.tensorflow.Tensor<?> tensor) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensorArray_Double(tensor.expect(Double.class), tensor.shape());
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensorArray_Float(tensor.expect(Float.class), tensor.shape());
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  public static Tensor getTensor(org.tensorflow.Tensor<?> tensor) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensor_Double(tensor.expect(Double.class), tensor.shape());
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensor_Float(tensor.expect(Float.class), tensor.shape());
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(Tensor data) {
    Tensor invertDimensions;
    double[] buffer;
    if(INVERT_RANKS) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    @NotNull org.tensorflow.@NotNull Tensor<Float> tensor = org.tensorflow.Tensor.create(Util.toLong(data.getDimensions()), FloatBuffer.wrap(Util.getFloats(buffer)));
    if(null != invertDimensions) invertDimensions.freeRef();
    return tensor;
  }

  @NotNull
  public static org.tensorflow.Tensor<Float> getFloatTensor(TensorList data) {
    double[] buffer = getDoubles(data);
    long[] shape = LongStream.concat(
        LongStream.of(data.length()),
        Arrays.stream(data.getDimensions()).mapToLong(x -> x)
    ).toArray();
    @NotNull org.tensorflow.@NotNull Tensor<Float> tensor = org.tensorflow.Tensor.create(shape, FloatBuffer.wrap(Util.getFloats(buffer)));
    RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    return tensor;
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(Tensor data) {
    double[] buffer;
    Tensor invertDimensions;
    if(INVERT_RANKS) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    org.tensorflow.Tensor<Double> tensor = org.tensorflow.Tensor.create(Util.toLong(data.getDimensions()), DoubleBuffer.wrap(buffer));
    if(null != invertDimensions) invertDimensions.freeRef();
    return tensor;
  }

  @NotNull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(TensorList data) {
    double[] buffer = getDoubles(data);
    long[] shape = LongStream.concat(
        LongStream.of(data.length()),
        Arrays.stream(data.getDimensions()).mapToLong(x -> x)
    ).toArray();
    org.tensorflow.Tensor<Double> tensor = org.tensorflow.Tensor.create(shape, DoubleBuffer.wrap(buffer));
    RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    return tensor;
  }


}
