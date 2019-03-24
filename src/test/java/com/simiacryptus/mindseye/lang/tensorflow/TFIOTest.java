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
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;

public class TFIOTest {

  private final double tol = 1e-4;

  private static void assertEquals(Tensor a, Tensor b, double tol) {
    assertArrayEquals(a.getDimensions(), b.getDimensions());
    assertArrayEquals(a.getData(), b.getData(), tol);
  }

  private static void assertEquals(TensorList a, TensorList b, double tol) {
    Assert.assertEquals(a.length(), b.length());
    for (int i = 0; i < a.length(); i++) {
      assertEquals(a.get(i), b.get(i), tol);
    }
  }

  @Test
  public void testTensor() {
    test(new Tensor(7).randomize(1.0));
    test(new Tensor(3, 2).randomize(1.0));
    test(new Tensor(3, 5, 2).randomize(1.0));
    test(new Tensor(5, 3, 2, 1).randomize(1.0));
  }

  public void test(Tensor tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor);
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor);
    assertArrayEquals(Arrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), doubleTensor.shape());
    assertEquals(tensor, TFIO.getTensor(doubleTensor), tol);
    assertArrayEquals(Arrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), floatTensor.shape());
    assertEquals(tensor, TFIO.getTensor(floatTensor), tol);
  }

  @Test
  public void testTensorList() {
    test(newTensorList(4, 7));
    test(newTensorList(4, 5, 2));
    test(newTensorList(4, 3, 3, 2));
  }

  public TensorArray newTensorList(int length, int... ints) {
    return TensorArray.wrap(IntStream.range(0, length).mapToObj(i -> new Tensor(ints).randomize(1.0)).toArray(i -> new Tensor[i]));
  }

  public void test(TensorList tensor) {
    org.tensorflow.Tensor<Double> doubleTensor = TFIO.getDoubleTensor(tensor);
    org.tensorflow.Tensor<Float> floatTensor = TFIO.getFloatTensor(tensor);
    Assert.assertEquals(tensor.length(), doubleTensor.shape()[0]);
    assertArrayEquals(Arrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), Arrays.stream(doubleTensor.shape()).skip(1).toArray());
    assertEquals(tensor, TFIO.getTensorList(doubleTensor), tol);
    Assert.assertEquals(tensor.length(), floatTensor.shape()[0]);
    assertArrayEquals(Arrays.stream(tensor.getDimensions()).mapToLong(x -> x).toArray(), Arrays.stream(floatTensor.shape()).skip(1).toArray());
    assertEquals(tensor, TFIO.getTensorList(floatTensor), tol);
  }
}
