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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class MaxPoolLayerTest extends RawTFLayerTestBase {

  @Nullable
  public static @SuppressWarnings("unused")
  MaxPoolLayerTest[] addRefs(@Nullable MaxPoolLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayerTest::addRef)
        .toArray((x) -> new MaxPoolLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MaxPoolLayerTest[][] addRefs(@Nullable MaxPoolLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayerTest::addRefs)
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MaxPoolLayerTest addRef() {
    return (MaxPoolLayerTest) super.addRef();
  }

  public static class Test0 extends MaxPoolLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Test0[] addRefs(@Nullable Test0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Test0::addRef).toArray((x) -> new Test0[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Test0 addRef() {
      return (Test0) super.addRef();
    }

    @Nonnull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(1);
      maxPoolLayer.setStrideY(1);
      return maxPoolLayer;
    }

  }

  public static class Test1 extends MaxPoolLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Test1[] addRefs(@Nullable Test1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Test1::addRef).toArray((x) -> new Test1[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Test1 addRef() {
      return (Test1) super.addRef();
    }

    @Nonnull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(2);
      maxPoolLayer.setStrideY(2);
      return maxPoolLayer;
    }

  }

  public static class Test2 extends MaxPoolLayerTest {
    @Nullable
    public static @SuppressWarnings("unused")
    Test2[] addRefs(@Nullable Test2[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Test2::addRef).toArray((x) -> new Test2[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Test2 addRef() {
      return (Test2) super.addRef();
    }

    @Nonnull
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
