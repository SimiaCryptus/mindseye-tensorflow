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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.LRNLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.tensorflow.MatMulLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayerBase;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.tensorflow.GraphModel;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.AttrValue;

import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.IntStream;

public class TFConverter {

  @NotNull
  public FullyConnectedLayer getFCLayer(MatMulLayer matMulLayer) {
    Tensor weights = matMulLayer.getWeights().get("weights");
    int[] intputDims = matMulLayer.getIntputDims();
    int[] outputDims = matMulLayer.getOutputDims();

    int[] tfView = Streams.concat(
        Arrays.stream(outputDims),
        IntStream.range(0, intputDims.length)
            .map(i -> (intputDims.length - 1) - i)
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
  public DAGNetwork convert(TFLayerBase tfLayer) {
    final PipelineNetwork converted = new PipelineNetwork(1);
    getNode(
        tfLayer.getOutputNode(),
        converted,
        new GraphModel(tfLayer.constGraph().toByteArray()),
        new HashMap<>()
    );
    return converted;
  }

  protected DAGNode getNode(String id, PipelineNetwork network, GraphModel tfModel, HashMap<String, DAGNode> map) {
    GraphModel.GraphNode graphNode = tfModel.getChild(id);
    assert null != graphNode;
    return map.computeIfAbsent(id, uuid -> {
      if (graphNode.getOp().equals("Conv2D")) {
        return network.add(
            getConv2D(graphNode),
            getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
      } else if (graphNode.getOp().equals("BiasAdd")) {
        return network.add(
            getBiasAdd(graphNode),
            getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
      } else if (graphNode.getOp().equals("Relu")) {
        return network.add(
            new ActivationLayer(ActivationLayer.Mode.RELU),
            getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
      } else if (graphNode.getOp().equals("LRN")) {
        return network.add(
            new LRNLayer(3),
            getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
      } else if (graphNode.getOp().equals("MaxPool")) {
        return network.add(
            new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max),
            getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
      } else if (graphNode.getOp().equals("Placeholder")) {
        return network.getInput(0);
      } else {
        throw new IllegalArgumentException(graphNode.getOp());
      }
    });
  }

  protected ImgBandBiasLayer getBiasAdd(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    double[] data = dataNode.getData();
    //Doubles.reverse(data);
    return new ImgBandBiasLayer(data.length).set(new Tensor(data, new int[]{data.length}));
  }

  protected Layer getConv2D(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    int[] kernelDims = Arrays.stream(dataNode.getShape()).mapToInt(x -> (int) x).toArray();
    double[] data = dataNode.getData();
    if (kernelDims.length == 0) kernelDims = new int[]{data.length};
    Tensor sourceKernel = new Tensor(data, new int[]{
        kernelDims[3],
        kernelDims[2],
        kernelDims[1],
        kernelDims[0]
    }).invertDimensionsAndFree();
    int[] sourceKernelDimensions = sourceKernel.getDimensions();
    SimpleConvolutionLayer convolutionLayer = new SimpleConvolutionLayer(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2] * sourceKernelDimensions[3]);
    Tensor targetKernel = new Tensor(
        sourceKernelDimensions[0],
        sourceKernelDimensions[1],
        sourceKernelDimensions[2],
        sourceKernelDimensions[3]
    );
//    ConvolutionLayer convolutionLayer = new ConvolutionLayer(sourceDims[0], sourceDims[1], sourceDims[2], sourceDims[3]);
    sourceKernel.coordStream(false).forEach(c -> {
      int[] coord = c.getCoords();
      targetKernel.set(
          (sourceKernelDimensions[0] - 1) - coord[0],
          (sourceKernelDimensions[1] - 1) - coord[1],
          coord[2],
          coord[3],
          sourceKernel.get(c)
      );
    });
    convolutionLayer.getKernel().set(targetKernel);
    targetKernel.freeRef();
    sourceKernel.freeRef();
    AttrValue stridesArr = graphNode.getNodeDef().getAttrMap().get("strides");
    if (null != stridesArr) {
      int[] strides = stridesArr.getList().getIList().stream().mapToInt(x -> Math.toIntExact(x)).toArray();
      int strideX = strides[1];
      int strideY = strides[2];
      if (strideX > 1 || strideY > 1) {
        convolutionLayer.setStrideX(strideX);
        convolutionLayer.setStrideY(strideY);
        return convolutionLayer;
      } else {
        return convolutionLayer.explode();
      }
    } else {
      return convolutionLayer.explode();
    }
  }

}
