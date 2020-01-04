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

public @com.simiacryptus.ref.lang.RefAware
class LRNLayerTest extends RawTFLayerTestBase {

  private final TFLayerBase tfLayer = createTFLayer();

  public LRNLayerTest() {
    validateDifferentials = false;
  }

  public static @SuppressWarnings("unused")
  LRNLayerTest[] addRefs(LRNLayerTest[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(LRNLayerTest::addRef)
        .toArray((x) -> new LRNLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  LRNLayerTest[][] addRefs(LRNLayerTest[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(LRNLayerTest::addRefs)
        .toArray((x) -> new LRNLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 3, 20}};
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LRNLayerTest addRef() {
    return (LRNLayerTest) super.addRef();
  }

  @NotNull
  protected TFLayerBase createTFLayer() {
    return new LRNLayer().setRadius(5).setAlpha(1e-4f).setBias(2);
  }

}
