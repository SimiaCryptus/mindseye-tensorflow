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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefAssert;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class TFIOTest {

  private final double tol = 1e-4;

  private static void assertEquals(Tensor a, Tensor b, double tol) {
    assertArrayEquals(a.getDimensions(), b.getDimensions());
    assertArrayEquals(a.getData(), b.getData(), tol);
    if (null != b)
      b.freeRef();
    if (null != a)
      a.freeRef();
  }

  private static void assertEquals(TensorList a, TensorList b, double tol) {
    RefAssert.assertEquals(a.length(), b.length());
    for (int i = 0; i < a.length(); i++) {
      assertEquals(a.get(i), b.get(i), tol);
    }
    if (null != b)
      b.freeRef();
    if (null != a)
      a.freeRef();
  }

  @Test
  public void testTensor() {
    Tensor temp_19_0001 = new Tensor(7);
    test(temp_19_0001.randomize(1.0));
    if (null != temp_19_0001)
      temp_19_0001.freeRef();
    Tensor temp_19_0002 = new Tensor(3, 2);
    test(temp_19_0002.randomize(1.0));
    if (null != temp_19_0002)
      temp_19_0002.freeRef();
    Tensor temp_19_0003 = new Tensor(3, 5, 2);
    test(temp_19_0003.randomize(1.0));
    if (null != temp_19_0003)
      temp_19_0003.freeRef();
    Tensor temp_19_0004 = new Tensor(5, 3, 2, 1);
    test(temp_19_0004.randomize(1.0));
    if (null != temp_19_0004)
      temp_19_0004.freeRef();
  }

  public void test(Tensor tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor == null ? null : tensor.addRef());
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor == null ? null : tensor.addRef());
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), doubleTensor.shape());
    assertEquals(tensor == null ? null : tensor.addRef(), TFIO.getTensor(doubleTensor), tol);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), floatTensor.shape());
    assertEquals(tensor == null ? null : tensor.addRef(), TFIO.getTensor(floatTensor), tol);
    if (null != tensor)
      tensor.freeRef();
  }

  @Test
  public void testTensorList() {
    test(newTensorList(4, 7));
    test(newTensorList(4, 5, 2));
    test(newTensorList(4, 3, 3, 2));
  }

  public TensorArray newTensorList(int length, int... ints) {
    return new TensorArray(RefIntStream.range(0, length).mapToObj(i -> {
      return new Tensor(ints).randomize(1.0);
    }).toArray(i -> new Tensor[i]));
  }

  public void test(TensorList tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor == null ? null : tensor.addRef());
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor == null ? null : tensor.addRef());
    RefAssert.assertEquals(tensor.length(), doubleTensor.shape()[0]);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(),
        RefArrays.stream(doubleTensor.shape()).skip(1).toArray());
    assertEquals(tensor == null ? null : tensor.addRef(), TFIO.getTensorList(doubleTensor), tol);
    RefAssert.assertEquals(tensor.length(), floatTensor.shape()[0]);
    assertArrayEquals(RefArrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(),
        RefArrays.stream(floatTensor.shape()).skip(1).toArray());
    assertEquals(tensor == null ? null : tensor.addRef(), TFIO.getTensorList(floatTensor), tol);
    if (null != tensor)
      tensor.freeRef();
  }
}
