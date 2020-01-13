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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefMap;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public class BiasLayerTest extends RawTFLayerTestBase {

  public static @SuppressWarnings("unused") BiasLayerTest[] addRefs(BiasLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasLayerTest::addRef)
        .toArray((x) -> new BiasLayerTest[x]);
  }

  public static @SuppressWarnings("unused") BiasLayerTest[][] addRefs(BiasLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasLayerTest::addRefs)
        .toArray((x) -> new BiasLayerTest[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 3, 3 } };
  }

  @NotNull
  public BiasLayer createTFLayer() {
    BiasLayer biasLayer = new BiasLayer(3, 3);
    RefMap<String, Tensor> temp_28_0001 = biasLayer.getWeights();
    Tensor temp_28_0002 = temp_28_0001.get("bias");
    RefUtil.freeRef(temp_28_0002.setByCoord(c -> Math.random()));
    if (null != temp_28_0002)
      temp_28_0002.freeRef();
    if (null != temp_28_0001)
      temp_28_0001.freeRef();
    return biasLayer;
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BiasLayerTest addRef() {
    return (BiasLayerTest) super.addRef();
  }

}
