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

import com.simiacryptus.ref.lang.RefAware;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public class LRNLayerTest extends RawTFLayerTestBase {

  private final TFLayerBase tfLayer = createTFLayer();

  public LRNLayerTest() {
    validateDifferentials = false;
  }

  public static @SuppressWarnings("unused") LRNLayerTest[] addRefs(LRNLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayerTest::addRef).toArray((x) -> new LRNLayerTest[x]);
  }

  public static @SuppressWarnings("unused") LRNLayerTest[][] addRefs(LRNLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayerTest::addRefs)
        .toArray((x) -> new LRNLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 2, 3, 20 } };
  }

  public @SuppressWarnings("unused") void _free() {
    if (null != tfLayer)
      tfLayer.freeRef();
  }

  public @Override @SuppressWarnings("unused") LRNLayerTest addRef() {
    return (LRNLayerTest) super.addRef();
  }

  @NotNull
  protected TFLayerBase createTFLayer() {
    LRNLayer temp_21_0002 = new LRNLayer();
    LRNLayer temp_21_0003 = temp_21_0002.setRadius(5);
    LRNLayer temp_21_0004 = temp_21_0003.setAlpha(1e-4f);
    LRNLayer temp_21_0001 = temp_21_0004.setBias(2);
    if (null != temp_21_0004)
      temp_21_0004.freeRef();
    if (null != temp_21_0003)
      temp_21_0003.freeRef();
    if (null != temp_21_0002)
      temp_21_0002.freeRef();
    return temp_21_0001;
  }

}
