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
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.util.TFConverter;

import javax.annotation.Nonnull;
import java.util.Random;


public class MatMulLayerTest extends LayerTestBase {

  private final int[] inputDim = {2, 2};

  //  @Override
//  public Tensor[] randomize(@Nonnull int[][] inputDims) {
//    Random random = new Random();
//    return Arrays.stream(inputDims).map(dim -> {
//      Tensor tensor = new Tensor(dim);
//      tensor.set(random.nextInt(tensor.length()), 1);
//      return tensor;
//    }).toArray(i -> new Tensor[i]);
//  }
  private final MatMulLayer matMulLayer = new MatMulLayer(inputDim, new int[]{2});

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        inputDim
    };
  }

  @Override
  public Layer getReferenceLayer() {
    return new TFConverter().getFCLayer(matMulLayer);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return matMulLayer.copy();
  }


}
