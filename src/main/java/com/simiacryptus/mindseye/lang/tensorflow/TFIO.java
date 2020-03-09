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
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.tensorflow.DataType;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.stream.Stream;

public class TFIO {

  @Nonnull
  public static TensorArray getTensorList(@Nonnull org.tensorflow.Tensor<?> tensor) {
    return getTensorList(tensor, true);
  }

  @Nonnull
  public static TensorArray getTensorList(@Nonnull org.tensorflow.Tensor<?> tensor, boolean invertRanks) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensorArray_Double(tensor.expect(Double.class), tensor.shape(), invertRanks);
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensorArray_Float(tensor.expect(Float.class), tensor.shape(), invertRanks);
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  @Nonnull
  public static Tensor getTensor(@Nonnull org.tensorflow.Tensor<?> tensor) {
    return getTensor(tensor, true);
  }

  @Nonnull
  public static Tensor getTensor(@Nonnull org.tensorflow.Tensor<?> tensor, boolean invertRanks) {
    if (tensor.dataType() == DataType.DOUBLE) {
      return getTensor_Double(tensor.expect(Double.class), tensor.shape(), invertRanks);
    } else if (tensor.dataType() == DataType.FLOAT) {
      return getTensor_Float(tensor.expect(Float.class), tensor.shape(), invertRanks);
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  @Nonnull
  public static org.tensorflow.Tensor<Float> getFloatTensor(@Nullable Tensor data) {
    org.tensorflow.Tensor<Float> temp_03_0006 = getFloatTensor(data == null ? null : data.addRef(), true);
    if (null != data)
      data.freeRef();
    return temp_03_0006;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Float> getFloatTensor(@Nonnull Tensor data, boolean invertRanks) {
    Tensor invertDimensions;
    double[] buffer;
    if (invertRanks) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    org.tensorflow.Tensor<Float> tfTensor = org.tensorflow.Tensor.create(
        Util.toLong(data.getDimensions()),
        FloatBuffer.wrap(Util.getFloats(buffer))
    );
    if (null != invertDimensions)
      invertDimensions.freeRef();
    data.freeRef();
    return tfTensor;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Float> getFloatTensor(@Nullable TensorList data) {
    return getFloatTensor(data, true);
  }

  @Nonnull
  public static org.tensorflow.Tensor<Float> getFloatTensor(@Nullable TensorList data, boolean invertRanks) {
    long[] shape = RefLongStream.concat(
        RefLongStream.of(data.length()),
        RefArrays.stream(data.getDimensions()).mapToLong(x -> x)
    ).toArray();
    double[] buffer = getDoubles(data, invertRanks);
    @Nonnull org.tensorflow.Tensor<Float> tensor = org.tensorflow.Tensor.create(shape,
        FloatBuffer.wrap(Util.getFloats(buffer)));
    RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    return tensor;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(@Nullable Tensor data) {
    org.tensorflow.Tensor<Double> temp_03_0009 = getDoubleTensor(data == null ? null : data.addRef(), true);
    if (null != data)
      data.freeRef();
    return temp_03_0009;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(@Nonnull Tensor data, boolean invertRanks) {
    double[] buffer;
    Tensor invertDimensions;
    if (invertRanks) {
      invertDimensions = data.invertDimensions();
      buffer = invertDimensions.getData();
    } else {
      invertDimensions = null;
      buffer = data.getData();
    }
    org.tensorflow.Tensor<Double> tfTensor = org.tensorflow.Tensor.create(
        Util.toLong(data.getDimensions()),
        DoubleBuffer.wrap(buffer)
    );
    data.freeRef();
    if (null != invertDimensions)
      invertDimensions.freeRef();
    return tfTensor;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(@Nullable TensorList data) {
    org.tensorflow.Tensor<Double> temp_03_0011 = getDoubleTensor(data == null ? null : data.addRef(), true);
    if (null != data)
      data.freeRef();
    return temp_03_0011;
  }

  @Nonnull
  public static org.tensorflow.Tensor<Double> getDoubleTensor(@Nullable TensorList data, boolean invertRanks) {
    long[] shape = RefLongStream.concat(
        RefLongStream.of(data.length()),
        RefArrays.stream(data.getDimensions()).mapToLong(x -> x)
    ).toArray();
    double[] buffer = getDoubles(data, invertRanks);
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

  private static Object createFloatArray(@Nonnull long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.FLOATS.obtain(shape[0]);
    } else if (shape.length == 2) {
      return RefIntStream.range(0, (int) shape[0]).mapToObj(i -> new float[(int) shape[1]])
          .toArray(s -> new float[s][]);
    } else if (shape.length == 3) {
      return RefIntStream.range(0, (int) shape[0]).mapToObj(i -> RefIntStream.range(0, (int) shape[1])
          .mapToObj(j -> new float[(int) shape[2]]).toArray(s -> new float[s][])).toArray(s -> new float[s][][]);
    } else if (shape.length == 4) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream
              .range(0, (int) shape[1]).mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> new float[(int) shape[3]]).toArray(s -> new float[s][]))
              .toArray(s -> new float[s][][]))
          .toArray(s -> new float[s][][][]);
    } else if (shape.length == 5) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(i -> RefIntStream.range(0, (int) shape[1])
              .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
                  .mapToObj(k -> RefIntStream.range(0, (int) shape[3]).mapToObj(l -> new float[(int) shape[4]])
                      .toArray(s -> new float[s][]))
                  .toArray(s -> new float[s][][]))
              .toArray(s -> new float[s][][][]))
          .toArray(s -> new float[s][][][][]);
    } else if (shape.length == 6) {
      return RefIntStream.range(0, (int) shape[0]).mapToObj(i -> RefIntStream.range(0, (int) shape[1])
          .mapToObj(j -> RefIntStream.range(0, (int) shape[2])
              .mapToObj(k -> RefIntStream.range(0, (int) shape[3])
                  .mapToObj(l -> RefIntStream.range(0, (int) shape[4]).mapToObj(m -> new float[(int) shape[5]])
                      .toArray(s -> new float[s][]))
                  .toArray(s -> new float[s][][]))
              .toArray(s -> new float[s][][][]))
          .toArray(s -> new float[s][][][][])).toArray(s -> new float[s][][][][][]);
    } else {
      throw new RuntimeException("Rank " + shape.length);
    }
  }

  private static Object createDoubleArray(@Nonnull long[] shape) {
    if (shape.length == 1) {
      return RecycleBin.DOUBLES.obtain(shape[0]);
    } else if (shape.length == 2) {
      return RefIntStream.range(0, (int) shape[0]).mapToObj(i -> new double[(int) shape[1]])
          .toArray(s -> new double[s][]);
    } else if (shape.length == 3) {
      return RefIntStream.range(0, (int) shape[0]).mapToObj(i -> RefIntStream.range(0, (int) shape[1])
          .mapToObj(j -> new double[(int) shape[2]]).toArray(s -> new double[s][])).toArray(s -> new double[s][][]);
    } else if (shape.length == 4) {
      return RefIntStream.range(0, (int) shape[0])
          .mapToObj(
              i -> RefIntStream.range(0, (int) shape[1])
                  .mapToObj(j -> RefIntStream.range(0, (int) shape[2]).mapToObj(k -> new double[(int) shape[3]])
                      .toArray(s -> new double[s][]))
                  .toArray(s -> new double[s][][]))
          .toArray(s -> new double[s][][][]);
    } else {
      throw new RuntimeException("Rank " + shape.length);
    }
  }

  @Nonnull
  private static RefDoubleStream flattenDoubles(Object obj) {
    if (obj instanceof double[]) {
      return RefArrays.stream((double[]) obj);
    } else if (obj instanceof Double) {
      return RefDoubleStream.of((double) obj);
    } else {
      return RefArrays.stream((Object[]) obj).flatMapToDouble(obj1 -> flattenDoubles(obj1));
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

  private static double[] getDoubles(@Nonnull TensorList data, boolean invertRanks) {
    double[] buffer = RecycleBin.DOUBLES.obtain(data.length() * Tensor.length(data.getDimensions()));
    DoubleBuffer inputBuffer = DoubleBuffer.wrap(buffer);
    if (invertRanks) {
      data.stream().map(tensor -> {
        Tensor invertDimensions = tensor.invertDimensions();
        tensor.freeRef();
        return invertDimensions;
      }).forEach(tensor -> {
        inputBuffer.put(tensor.getData());
        tensor.freeRef();
      });
    } else {
      data.stream().forEach(tensor -> {
        inputBuffer.put(tensor.getData());
        tensor.freeRef();
      });
    }
    data.freeRef();
    return buffer;
  }

  @Nonnull
  private static TensorArray getTensorArray_Float(@Nonnull org.tensorflow.Tensor<Float> tensor, @Nonnull long[] shape,
                                                  boolean invertRanks) {
    float[] doubles = getFloats(tensor);
    int[] dims = RefArrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = new TensorArray(RefIntStream.range(0, batches).mapToObj(i -> {
      int offset = i * Tensor.length(dims);
      if (invertRanks) {
        Tensor returnValue = new Tensor(Tensor.reverse(dims));
        returnValue.set(j -> doubles[j + offset]);
        Tensor invertDimensions = returnValue.invertDimensions();
        returnValue.freeRef();
        return invertDimensions;
      } else {
        Tensor returnValue = new Tensor(dims);
        returnValue.set(j -> doubles[j + offset]);
        return returnValue;
      }
    }).toArray(i -> new Tensor[i]));
    RecycleBin.FLOATS.recycle(doubles, doubles.length);
    return resultData;
  }

  @Nonnull
  private static Tensor getTensor_Float(@Nonnull org.tensorflow.Tensor<Float> tensor, @Nonnull long[] shape, boolean invertRanks) {
    if (0 == tensor.numElements())
      return new Tensor(RefArrays.stream(shape).mapToInt(x -> (int) x).toArray());
    float[] doubles = getFloats(tensor);
    int[] dims = RefArrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if (invertRanks) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      returnValue.set(j -> doubles[j]);
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      Tensor invertDimensions = returnValue.invertDimensions();
      returnValue.freeRef();
      return invertDimensions;
    } else {
      Tensor returnValue = new Tensor(dims);
      returnValue.set(j -> doubles[j]);
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return returnValue;
    }
  }

  @Nonnull
  private static TensorArray getTensorArray_Double(@Nonnull org.tensorflow.Tensor<Double> tensor, @Nonnull long[] shape,
                                                   boolean invertRanks) {
    double[] doubles = getDoubles(tensor);
    int[] dims = RefArrays.stream(shape).skip(1).mapToInt(x -> (int) x).toArray();
    int batches = (int) shape[0];
    TensorArray resultData = new TensorArray(RefIntStream.range(0, batches).mapToObj(i -> {
      if (invertRanks) {
        Tensor returnValue = new Tensor(Tensor.reverse(dims));
        RefSystem.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0,
            returnValue.length());
        Tensor invertDimensions = returnValue.invertDimensions();
        returnValue.freeRef();
        return invertDimensions;
      } else {
        Tensor returnValue = new Tensor(dims);
        RefSystem.arraycopy(doubles, i * returnValue.length(), returnValue.getData(), 0,
            returnValue.length());
        return returnValue;
      }
    }).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(doubles, doubles.length);
    return resultData;
  }

  @Nonnull
  private static Tensor getTensor_Double(@Nonnull org.tensorflow.Tensor<Double> tensor, @Nonnull long[] shape, boolean invertRanks) {
    double[] doubles = getDoubles(tensor);
    int[] dims = RefArrays.stream(shape).mapToInt(x -> (int) x).toArray();
    if (invertRanks) {
      Tensor returnValue = new Tensor(Tensor.reverse(dims));
      RefSystem.arraycopy(doubles, 0, returnValue.getData(), 0, returnValue.length());
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      Tensor invertDimensions = returnValue.invertDimensions();
      returnValue.freeRef();
      return invertDimensions;
    } else {
      Tensor returnValue = new Tensor(dims);
      RefSystem.arraycopy(doubles, 0, returnValue.getData(), 0, returnValue.length());
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      return returnValue;
    }
  }

  private static double[] getDoubles(@Nonnull org.tensorflow.Tensor<Double> result) {
    Object deepArray = result.copyTo(createDoubleArray(result.shape()));
    double[] doubles = flattenDoubles(deepArray).toArray();
    free(deepArray);
    return doubles;
  }

  @Nonnull
  private static float[] getFloats(@Nonnull org.tensorflow.Tensor<Float> result) {
    if (0 == result.numElements())
      return new float[]{};
    Object deepArray = result.copyTo(createFloatArray(result.shape()));
    double[] doubles = flattenFloats(deepArray).mapToDouble(x -> x).toArray();
    free(deepArray);
    return Util.getFloats(doubles);
  }

}
