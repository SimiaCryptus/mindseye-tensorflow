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

package com.simiacryptus.mindseye.util;

import com.google.common.collect.Streams;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.tensorflow.MatMulLayer;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.stream.IntStream;

public class TFConverter {

  @NotNull
  public static FullyConnectedLayer getFCLayer(MatMulLayer matMulLayer) {
    Tensor weights = matMulLayer.getWeights().get("weights");
    int[] intputDims = matMulLayer.getIntputDims();
    int[] outputDims = matMulLayer.getOutputDims();

    int[] tfView = Streams.concat(
        Arrays.stream(outputDims),
        IntStream.range(0, intputDims.length)
            .map(i->(intputDims.length-1)-i)
            .map(i -> intputDims[i])
    ).toArray();
    int[] tfPermute = Streams.concat(
        IntStream.range(0, intputDims.length).map(i -> outputDims.length + ((intputDims.length - 1) - i)),
        IntStream.range(0, outputDims.length)
    ).toArray();
    Tensor rearranged = weights
        .reshapeCast(tfView)
        .permuteDimensionsAndFree(tfPermute);

    FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(intputDims, outputDims);
    fullyConnectedLayer.getWeights().set(rearranged);
    rearranged.freeRef();
    return fullyConnectedLayer;
  }

  @NotNull
  public static SimpleConvolutionLayer getSimpleConvolutionLayer(Tensor sourceKernel) {
    int[] kernelDims = sourceKernel.getDimensions();
    SimpleConvolutionLayer simpleConvolutionLayer = new SimpleConvolutionLayer(kernelDims[0], kernelDims[1], kernelDims[2] * kernelDims[3]).setPaddingXY(0, 0);
    Tensor targetKernel = sourceKernel.copy();
    int[] sourceDims = sourceKernel.getDimensions();
    sourceKernel.coordStream(false).forEach(c -> {
      int[] sourceCoords = c.getCoords();
      targetKernel.set(
          sourceDims[0] - (1 + sourceCoords[0]),
          sourceDims[1] - (1 + sourceCoords[1]),
          sourceCoords[2],
          sourceCoords[3],
          sourceKernel.get(c)
      );
    });
    simpleConvolutionLayer.kernel.set(targetKernel);
    targetKernel.freeRef();
    return simpleConvolutionLayer;
  }
}
