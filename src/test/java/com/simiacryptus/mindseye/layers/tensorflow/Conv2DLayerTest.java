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

import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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
  public Tensor[] randomize(@Nonnull int[][] inputDims) {
    Random random = new Random();
    return Arrays.stream(inputDims).map(dim -> {
      Tensor tensor = new Tensor(dim);
      tensor.set(random.nextInt(tensor.length()), 1);
      //tensor.set(() -> random());
      return tensor;
    }).toArray(i -> new Tensor[i]);
  }

  @Override
  public Layer getReferenceLayer() {
//    if(1==1) return null;
    Tensor kernel = layer.getWeights().get("kernel");
    int[] kernelDims = kernel.getDimensions();
    SimpleConvolutionLayer simpleConvolutionLayer = new SimpleConvolutionLayer(kernelDims[0], kernelDims[1], kernelDims[2] * kernelDims[3]).setPaddingXY(0, 0);
    Tensor targetKernel = kernel.reshapeCast(
        kernelDims[1],
        kernelDims[0],
        kernelDims[3],
        kernelDims[2]
    );
    Tensor sourceKernel = kernel; //.permuteDimensions(0,1,2,3);
    int[] sourceDims = sourceKernel.getDimensions();
    sourceKernel.coordStream(false).forEach(c -> {
      int[] sourceCoords = c.getCoords();
      targetKernel.set(
//          sourceDims[3] - (1+ sourceCoords[3]),
          sourceDims[1] - (1 + sourceCoords[1]),
          sourceDims[0] - (1 + sourceCoords[0]),
          sourceCoords[3],
          sourceCoords[2],
//          sourceDims[2] - (1+sourceCoords[2]),
          sourceKernel.get(c)
      );
    });
    simpleConvolutionLayer.kernel.set(targetKernel);
    targetKernel.freeRef();
    return simpleConvolutionLayer;
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  private final Conv2DLayer layer = getLayer();

  @NotNull
  private Conv2DLayer getLayer() {
    Conv2DLayer layer = new Conv2DLayer(1, 1, 2, 2);
    Tensor kernel = layer.getWeights().get("kernel");
    //kernel.setByCoord(c -> Math.random());
    kernel.set(0, 0, 0, 0, 1.0);
    kernel.set(0, 0, 1, 1, 1.0);
//    kernel.set(kernel.invertDimensions());
    //kernel.toJson(null,SerialPrecision.Double);
    return layer;
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer.copy();
  }


}
