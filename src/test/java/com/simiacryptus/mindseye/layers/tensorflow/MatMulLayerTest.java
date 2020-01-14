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

public class MatMulLayerTest extends RawTFLayerTestBase {

  private final int[] inputDim = {2, 2};

  @Nullable
  public static @SuppressWarnings("unused")
  MatMulLayerTest[] addRefs(@Nullable MatMulLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MatMulLayerTest::addRef)
        .toArray((x) -> new MatMulLayerTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MatMulLayerTest[][] addRefs(@Nullable MatMulLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MatMulLayerTest::addRefs)
        .toArray((x) -> new MatMulLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{inputDim};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MatMulLayerTest addRef() {
    return (MatMulLayerTest) super.addRef();
  }

  @Nonnull
  protected MatMulLayer createTFLayer() {
    return new MatMulLayer(inputDim, new int[]{2});
  }

}
