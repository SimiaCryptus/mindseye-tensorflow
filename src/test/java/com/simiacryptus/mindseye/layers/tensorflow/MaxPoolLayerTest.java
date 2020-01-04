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

package com.simiacryptus.mindseye.layers.tensorflow;

import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract @com.simiacryptus.ref.lang.RefAware
class MaxPoolLayerTest extends RawTFLayerTestBase {

  public static @SuppressWarnings("unused")
  MaxPoolLayerTest[] addRefs(MaxPoolLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayerTest::addRef)
        .toArray((x) -> new MaxPoolLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  MaxPoolLayerTest[][] addRefs(MaxPoolLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayerTest::addRefs)
        .toArray((x) -> new MaxPoolLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{5, 5, 1}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MaxPoolLayerTest addRef() {
    return (MaxPoolLayerTest) super.addRef();
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Test0 extends MaxPoolLayerTest {
    public static @SuppressWarnings("unused")
    Test0[] addRefs(Test0[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Test0::addRef).toArray((x) -> new Test0[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Test0 addRef() {
      return (Test0) super.addRef();
    }

    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(1);
      maxPoolLayer.setStrideY(1);
      return maxPoolLayer;
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Test1 extends MaxPoolLayerTest {
    public static @SuppressWarnings("unused")
    Test1[] addRefs(Test1[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Test1::addRef).toArray((x) -> new Test1[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Test1 addRef() {
      return (Test1) super.addRef();
    }

    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(2);
      maxPoolLayer.setStrideY(2);
      return maxPoolLayer;
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class Test2 extends MaxPoolLayerTest {
    public static @SuppressWarnings("unused")
    Test2[] addRefs(Test2[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(Test2::addRef).toArray((x) -> new Test2[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Test2 addRef() {
      return (Test2) super.addRef();
    }

    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(3);
      maxPoolLayer.setHeight(3);
      maxPoolLayer.setStrideX(2);
      maxPoolLayer.setStrideY(2);
      return maxPoolLayer;
    }

  }

}
