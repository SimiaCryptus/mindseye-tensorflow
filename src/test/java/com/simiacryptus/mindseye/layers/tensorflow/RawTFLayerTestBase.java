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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class RawTFLayerTestBase extends TFLayerTestBase {

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return new TFConverter().convert(getTfLayer());
  }

  public static @SuppressWarnings("unused") RawTFLayerTestBase[] addRefs(RawTFLayerTestBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RawTFLayerTestBase::addRef)
        .toArray((x) -> new RawTFLayerTestBase[x]);
  }

  public static @SuppressWarnings("unused") RawTFLayerTestBase[][] addRefs(RawTFLayerTestBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RawTFLayerTestBase::addRefs)
        .toArray((x) -> new RawTFLayerTestBase[x][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return getTfLayer();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") RawTFLayerTestBase addRef() {
    return (RawTFLayerTestBase) super.addRef();
  }

}
