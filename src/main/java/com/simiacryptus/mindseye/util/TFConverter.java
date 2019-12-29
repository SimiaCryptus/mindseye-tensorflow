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
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.tensorflow.MatMulLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayerBase;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TFConverter {

  public static List<TFLayer> getLayers(ImageNetworkPipeline pipeline) {
    return IntStream.range(0, pipeline.graphDefs.size()).mapToObj(i -> getLayer(pipeline, i))
        .collect(Collectors.toList());
  }

  @NotNull
  public static TFLayer getLayer(ImageNetworkPipeline pipeline, int i) {
    GraphDef graphDef = pipeline.graphDefs.get(i);
    String output = pipeline.nodeIds().get(i);
    String input = i == 0 ? "input" : pipeline.nodeIds().get(i - 1);
    return new TFLayer(graphDef.toByteArray(), new HashMap<>(), output, input).setFloat(true);
  }

  @NotNull
  public FullyConnectedLayer getFCLayer(MatMulLayer matMulLayer) {
    Tensor weights = matMulLayer.getWeights().get("weights");
    int[] intputDims = matMulLayer.getIntputDims();
    int[] outputDims = matMulLayer.getOutputDims();

    int[] tfView = Streams
        .concat(Arrays.stream(outputDims),
            IntStream.range(0, intputDims.length).map(i -> (intputDims.length - 1) - i).map(i -> intputDims[i]))
        .toArray();
    int[] tfPermute = Streams
        .concat(IntStream.range(0, intputDims.length).map(i -> outputDims.length + ((intputDims.length - 1) - i)),
            IntStream.range(0, outputDims.length))
        .toArray();
    Tensor rearranged = weights.reshapeCast(tfView).permuteDimensions(tfPermute);

    FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(intputDims, outputDims);
    fullyConnectedLayer.getWeights().set(rearranged);
    return fullyConnectedLayer;
  }

  @NotNull
  public PipelineNetwork convert(TFLayerBase tfLayer) {
    final PipelineNetwork converted = new PipelineNetwork(1);
    ConcurrentHashMap<String, DAGNode> nodes = new ConcurrentHashMap<>();
    getNode(tfLayer.getOutputNode(), converted, new GraphModel(tfLayer.constGraph().toByteArray()), nodes);
    return converted;
  }

  protected DAGNode getNode(String id, PipelineNetwork network, GraphModel tfModel,
                            ConcurrentHashMap<String, DAGNode> map) {
    try {
      if (!map.containsKey(id)) {
        DAGNode result;
        GraphModel.GraphNode graphNode = tfModel.getChild(id);
        assert null != graphNode;
        if (graphNode.getOp().equals("Conv2D")) {
          result = network.add(getConv2D(graphNode), getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
        } else if (graphNode.getOp().equals("BiasAdd")) {
          result = network.add(getBiasAdd(graphNode), getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
        } else if (graphNode.getOp().equals("Relu")) {
          result = network.add(new ActivationLayer(ActivationLayer.Mode.RELU), getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
        } else if (graphNode.getOp().equals("LRN")) {
          result = network.add(getLRNLayer(graphNode), getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
        } else if (graphNode.getOp().equals("MaxPool")) {
          result = network.add(getPoolingLayer(graphNode), getNode(graphNode.getInputKeys().get(0), network, tfModel, map));
        } else if (graphNode.getOp().equals("Concat")) {
          List<String> inputKeys = graphNode.getInputKeys();
          result = network.add(new ImgConcatLayer(), inputKeys.stream().skip(1)
              .map(inputKey -> getNode(inputKey, network, tfModel, map)).toArray(i -> new DAGNode[i]));
        } else if (graphNode.getOp().equals("Placeholder")) {
          result = network.getInput(0);
        } else {
          throw new IllegalArgumentException(graphNode.getOp());
        }
        if (!map.containsKey(id)) {
          map.put(id, result);
        }
      }
      return map.get(id);
    } catch (Throwable e) {
      throw new RuntimeException("Error converting " + id, e);
    }
  }

  @NotNull
  protected PoolingLayer getPoolingLayer(GraphModel.GraphNode graphNode) {
    PoolingLayer poolingLayer = new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max);
    Map<String, AttrValue> attrMap = graphNode.getNodeDef().getAttrMap();
    assert "SAME".equals(attrMap.get("padding").getS().toStringUtf8());
    AttrValue _ksize = attrMap.get("ksize");
    if (null != _ksize) {
      List<Long> ksize = _ksize.getList().getIList();
      poolingLayer.setWindowX(Math.toIntExact(ksize.get(1)));
      poolingLayer.setWindowY(Math.toIntExact(ksize.get(2)));
    }
    AttrValue _strides = attrMap.get("strides");
    if (null != _strides) {
      List<Long> strides = _strides.getList().getIList();
      poolingLayer.setStrideX(Math.toIntExact(strides.get(1)));
      poolingLayer.setStrideY(Math.toIntExact(strides.get(2)));
    }
    return poolingLayer;
  }

  protected ImgBandBiasLayer getBiasAdd(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    double[] data = dataNode.getData();
    Tensor tensor = new Tensor(data, data.length);
    return new ImgBandBiasLayer(data.length).set(tensor);
  }

  protected Layer getConv2D(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    int[] kernelDims = Arrays.stream(dataNode.getShape()).mapToInt(x -> (int) x).toArray();
    double[] data = dataNode.getData();
    if (kernelDims.length == 0)
      kernelDims = new int[]{data.length};
    Tensor sourceKernel = new Tensor(data, kernelDims[3], kernelDims[2], kernelDims[1], kernelDims[0])
        .invertDimensions();
    int[] sourceKernelDimensions = sourceKernel.getDimensions();
    //    ConvolutionLayer convolutionLayer = new ConvolutionLayer(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2], sourceKernelDimensions[3]);
    SimpleConvolutionLayer convolutionLayer = new SimpleConvolutionLayer(sourceKernelDimensions[0],
        sourceKernelDimensions[1], sourceKernelDimensions[2] * sourceKernelDimensions[3]);
    Tensor targetKernel = new Tensor(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2],
        sourceKernelDimensions[3]);
    sourceKernel.coordStream(false).forEach(c -> {
      int[] coord = c.getCoords();
      targetKernel.set((sourceKernelDimensions[0] - 1) - coord[0], (sourceKernelDimensions[1] - 1) - coord[1], coord[2],
          coord[3], sourceKernel.get(c));
    });
    convolutionLayer.getKernel().set(targetKernel);
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
        return convolutionLayer;
      }
    } else {
      return convolutionLayer;
    }
  }

  @NotNull
  private LRNLayer getLRNLayer(GraphModel.GraphNode graphNode) {
    Map<String, AttrValue> attrMap = graphNode.getNodeDef().getAttrMap();
    long depth_radius = attrMap.get("depth_radius").getI();
    float alpha = attrMap.get("alpha").getF();
    float bias = attrMap.get("bias").getF();
    float beta = attrMap.get("beta").getF();
    long width = depth_radius * 2 + 1;
    return new LRNLayer((int) width).setAlpha(alpha * width).setBeta(beta).setK(bias);
  }

}
