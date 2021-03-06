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
import com.simiacryptus.ref.wrappers.RefMap;

import javax.annotation.Nonnull;

public class BiasAddLayerTest extends RawTFLayerTestBase {

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 2, 3}};
  }

  public @Nonnull
  TFLayerBase createTFLayer() {
    BiasAddLayer biasLayer = new BiasAddLayer(3);
    RefMap<String, Tensor> temp_23_0001 = biasLayer.getWeights();
    assert temp_23_0001 != null;
    Tensor temp_23_0002 = temp_23_0001.get("bias");
    assert temp_23_0002 != null;
    temp_23_0002.setByCoord(c -> Math.random());
    temp_23_0002.freeRef();
    temp_23_0001.freeRef();
    return biasLayer;
  }

}
