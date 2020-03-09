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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefAssert;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import static org.junit.Assert.assertArrayEquals;

public class TFIOTest {

  private final double tol = 1e-4;

  private static void assertEquals(@Nonnull Tensor a, @Nonnull Tensor b, double tol) {
    assertArrayEquals(a.getDimensions(), b.getDimensions());
    assertArrayEquals(a.getData(), b.getData(), tol);
    b.freeRef();
    a.freeRef();
  }

  private static void assertEquals(@Nonnull TensorList a, @Nonnull TensorList b, double tol) {
    RefAssert.assertEquals(a.length(), b.length());
    for (int i = 0; i < a.length(); i++) {
      assertEquals(a.get(i), b.get(i), tol);
    }
    b.freeRef();
    a.freeRef();
  }

  @Test
  public void testTensor() {
    Tensor temp_19_0001 = new Tensor(7);
    temp_19_0001.randomize(1.0);
    test(temp_19_0001.addRef());
    temp_19_0001.freeRef();
    Tensor temp_19_0002 = new Tensor(3, 2);
    temp_19_0002.randomize(1.0);
    test(temp_19_0002.addRef());
    temp_19_0002.freeRef();
    Tensor temp_19_0003 = new Tensor(3, 5, 2);
    temp_19_0003.randomize(1.0);
    test(temp_19_0003.addRef());
    temp_19_0003.freeRef();
    Tensor temp_19_0004 = new Tensor(5, 3, 2, 1);
    temp_19_0004.randomize(1.0);
    test(temp_19_0004.addRef());
    temp_19_0004.freeRef();
  }

  public void test(@Nullable Tensor tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor == null ? null : tensor.addRef());
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor == null ? null : tensor.addRef());
    assert tensor != null;
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), doubleTensor.shape());
    assertEquals(tensor.addRef(), TFIO.getTensor(doubleTensor), tol);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), floatTensor.shape());
    assertEquals(tensor.addRef(), TFIO.getTensor(floatTensor), tol);
    tensor.freeRef();
  }

  @Test
  public void testTensorList() {
    test(newTensorList(4, 7));
    test(newTensorList(4, 5, 2));
    test(newTensorList(4, 3, 3, 2));
  }

  @Nonnull
  public TensorArray newTensorList(int length, int... ints) {
    return new TensorArray(RefIntStream.range(0, length).mapToObj(i -> {
      Tensor tensor = new Tensor(ints);
      tensor.randomize(1.0);
      return tensor;
    }).toArray(i -> new Tensor[i]));
  }

  public void test(@Nullable TensorList tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor == null ? null : tensor.addRef());
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor == null ? null : tensor.addRef());
    assert tensor != null;
    RefAssert.assertEquals(tensor.length(), (int) doubleTensor.shape()[0]);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(),
        RefArrays.stream(doubleTensor.shape()).skip(1).toArray());
    assertEquals(tensor.addRef(), TFIO.getTensorList(doubleTensor), tol);
    RefAssert.assertEquals(tensor.length(), (int) floatTensor.shape()[0]);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(),
        RefArrays.stream(floatTensor.shape()).skip(1).toArray());
    assertEquals(tensor.addRef(), TFIO.getTensorList(floatTensor), tol);
    tensor.freeRef();
  }
}
