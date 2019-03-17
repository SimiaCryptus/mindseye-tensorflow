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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.util.TFConverter;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Random;


public class Conv2DLayerTest extends LayerTestBase {

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {3, 3, 2}
    };
  }

  @Override
  public Layer getReferenceLayer() {
    return TFConverter.getSimpleConvolutionLayer(layer.getWeights().get("kernel"));
  }

  private final Conv2DLayer layer = getLayer();

  @NotNull
  private Conv2DLayer getLayer() {
    Conv2DLayer layer = new Conv2DLayer(3, 3, 2, 2);
    Tensor kernel = layer.getWeights().get("kernel");
    kernel.randomize(1.0);
    return layer;
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer.copy();
  }


}
