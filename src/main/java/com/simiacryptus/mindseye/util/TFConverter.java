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
import com.simiacryptus.mindseye.lang.Coordinate;
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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

public @RefAware
class TFConverter {

  public static RefList<TFLayer> getLayers(ImageNetworkPipeline pipeline) {
    return RefIntStream.range(0, pipeline.graphDefs.size()).mapToObj(i -> getLayer(pipeline, i))
        .collect(RefCollectors.toList());
  }

  @NotNull
  public static TFLayer getLayer(ImageNetworkPipeline pipeline, int i) {
    GraphDef graphDef = pipeline.graphDefs.get(i);
    String output = pipeline.nodeIds().get(i);
    String input = i == 0 ? "input" : pipeline.nodeIds().get(i - 1);
    TFLayer temp_05_0004 = new TFLayer(graphDef.toByteArray(),
        new RefHashMap<>(), output, input);
    TFLayer temp_05_0003 = temp_05_0004.setFloat(true);
    if (null != temp_05_0004)
      temp_05_0004.freeRef();
    return temp_05_0003;
  }

  @NotNull
  public FullyConnectedLayer getFCLayer(MatMulLayer matMulLayer) {
    RefMap<String, Tensor> temp_05_0010 = matMulLayer
        .getWeights();
    Tensor weights = temp_05_0010.get("weights");
    if (null != temp_05_0010)
      temp_05_0010.freeRef();
    int[] intputDims = matMulLayer.getIntputDims();
    int[] outputDims = matMulLayer.getOutputDims();

    if (null != matMulLayer)
      matMulLayer.freeRef();
    int[] tfView = Streams
        .concat(RefArrays.stream(outputDims),
            RefIntStream.range(0, intputDims.length).map(i -> (intputDims.length - 1) - i).map(i -> intputDims[i]))
        .toArray();
    int[] tfPermute = Streams
        .concat(RefIntStream.range(0, intputDims.length).map(i -> outputDims.length + ((intputDims.length - 1) - i)),
            RefIntStream.range(0, outputDims.length))
        .toArray();
    Tensor temp_05_0011 = weights.reshapeCast(tfView);
    Tensor rearranged = temp_05_0011.permuteDimensions(tfPermute);

    if (null != temp_05_0011)
      temp_05_0011.freeRef();
    if (null != weights)
      weights.freeRef();
    FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(intputDims, outputDims);
    Tensor temp_05_0012 = fullyConnectedLayer.getWeights();
    temp_05_0012.set(rearranged == null ? null : rearranged.addRef());
    if (null != temp_05_0012)
      temp_05_0012.freeRef();
    if (null != rearranged)
      rearranged.freeRef();
    return fullyConnectedLayer;
  }

  @NotNull
  public PipelineNetwork convert(TFLayerBase tfLayer) {
    final PipelineNetwork converted = new PipelineNetwork(1);
    RefConcurrentHashMap<String, DAGNode> nodes = new RefConcurrentHashMap<>();
    RefUtil
        .freeRef(getNode(tfLayer.getOutputNode(), converted == null ? null : converted.addRef(),
            new GraphModel(tfLayer.constGraph().toByteArray()), RefUtil.addRef(nodes)));
    if (null != tfLayer)
      tfLayer.freeRef();
    if (null != nodes)
      nodes.freeRef();
    return converted;
  }

  protected DAGNode getNode(String id, PipelineNetwork network, GraphModel tfModel,
                            RefConcurrentHashMap<String, DAGNode> map) {
    try {
      if (!map.containsKey(id)) {
        DAGNode result;
        GraphModel.GraphNode graphNode = tfModel.getChild(id);
        assert null != graphNode;
        if (graphNode.getOp().equals("Conv2D")) {
          result = network.add(getConv2D(graphNode), getNode(graphNode.getInputKeys().get(0),
              network == null ? null : network.addRef(), tfModel, RefUtil.addRef(map)));
        } else if (graphNode.getOp().equals("BiasAdd")) {
          result = network.add(getBiasAdd(graphNode), getNode(graphNode.getInputKeys().get(0),
              network == null ? null : network.addRef(), tfModel, RefUtil.addRef(map)));
        } else if (graphNode.getOp().equals("Relu")) {
          result = network.add(new ActivationLayer(ActivationLayer.Mode.RELU), getNode(graphNode.getInputKeys().get(0),
              network == null ? null : network.addRef(), tfModel, RefUtil.addRef(map)));
        } else if (graphNode.getOp().equals("LRN")) {
          result = network.add(getLRNLayer(graphNode), getNode(graphNode.getInputKeys().get(0),
              network == null ? null : network.addRef(), tfModel, RefUtil.addRef(map)));
        } else if (graphNode.getOp().equals("MaxPool")) {
          result = network.add(getPoolingLayer(graphNode), getNode(graphNode.getInputKeys().get(0),
              network == null ? null : network.addRef(), tfModel, RefUtil.addRef(map)));
        } else if (graphNode.getOp().equals("Concat")) {
          List<String> inputKeys = graphNode.getInputKeys();
          result = network.add(new ImgConcatLayer(),
              inputKeys.stream().skip(1).map(RefUtil.wrapInterface(
                  (Function<? super String, ? extends DAGNode>) inputKey -> getNode(
                      inputKey, network == null ? null : network.addRef(), tfModel,
                      RefUtil.addRef(map)),
                  RefUtil.addRef(map), network == null ? null : network.addRef()))
                  .toArray(i -> new DAGNode[i]));
        } else if (graphNode.getOp().equals("Placeholder")) {
          result = network.getInput(0);
        } else {
          throw new IllegalArgumentException(graphNode.getOp());
        }
        if (!map.containsKey(id)) {
          RefUtil.freeRef(map.put(id, result == null ? null : result.addRef()));
        }
        if (null != result)
          result.freeRef();
      }
      if (null != network)
        network.freeRef();
      DAGNode temp_05_0002 = map.get(id);
      if (null != map)
        map.freeRef();
      return temp_05_0002;
    } catch (Throwable e) {
      throw new RuntimeException("Error converting " + id, e);
    }
  }

  @NotNull
  protected PoolingLayer getPoolingLayer(GraphModel.GraphNode graphNode) {
    PoolingLayer temp_05_0005 = new PoolingLayer();
    PoolingLayer poolingLayer = temp_05_0005.setMode(PoolingLayer.PoolingMode.Max);
    if (null != temp_05_0005)
      temp_05_0005.freeRef();
    Map<String, AttrValue> attrMap = graphNode.getNodeDef().getAttrMap();
    assert "SAME".equals(attrMap.get("padding").getS().toStringUtf8());
    AttrValue _ksize = attrMap.get("ksize");
    if (null != _ksize) {
      List<Long> ksize = _ksize.getList().getIList();
      RefUtil.freeRef(poolingLayer.setWindowX(Math.toIntExact(ksize.get(1))));
      RefUtil.freeRef(poolingLayer.setWindowY(Math.toIntExact(ksize.get(2))));
    }
    AttrValue _strides = attrMap.get("strides");
    if (null != _strides) {
      List<Long> strides = _strides.getList().getIList();
      RefUtil.freeRef(poolingLayer.setStrideX(Math.toIntExact(strides.get(1))));
      RefUtil.freeRef(poolingLayer.setStrideY(Math.toIntExact(strides.get(2))));
    }
    return poolingLayer;
  }

  protected ImgBandBiasLayer getBiasAdd(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    double[] data = dataNode.getData();
    Tensor tensor = new Tensor(data, data.length);
    ImgBandBiasLayer temp_05_0006 = new ImgBandBiasLayer(data.length);
    ImgBandBiasLayer temp_05_0001 = temp_05_0006
        .set(tensor == null ? null : tensor.addRef());
    if (null != temp_05_0006)
      temp_05_0006.freeRef();
    if (null != tensor)
      tensor.freeRef();
    return temp_05_0001;
  }

  protected Layer getConv2D(GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    int[] kernelDims = RefArrays.stream(dataNode.getShape()).mapToInt(x -> (int) x).toArray();
    double[] data = dataNode.getData();
    if (kernelDims.length == 0)
      kernelDims = new int[]{data.length};
    Tensor temp_05_0007 = new Tensor(data, kernelDims[3], kernelDims[2], kernelDims[1],
        kernelDims[0]);
    Tensor sourceKernel = temp_05_0007.invertDimensions();
    if (null != temp_05_0007)
      temp_05_0007.freeRef();
    int[] sourceKernelDimensions = sourceKernel.getDimensions();
    //    ConvolutionLayer convolutionLayer = new ConvolutionLayer(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2], sourceKernelDimensions[3]);
    SimpleConvolutionLayer convolutionLayer = new SimpleConvolutionLayer(sourceKernelDimensions[0],
        sourceKernelDimensions[1], sourceKernelDimensions[2] * sourceKernelDimensions[3]);
    Tensor targetKernel = new Tensor(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2],
        sourceKernelDimensions[3]);
    sourceKernel.coordStream(false).forEach(RefUtil
        .wrapInterface((Consumer<? super Coordinate>) c -> {
          int[] coord = c.getCoords();
          targetKernel.set((sourceKernelDimensions[0] - 1) - coord[0], (sourceKernelDimensions[1] - 1) - coord[1],
              coord[2], coord[3], sourceKernel.get(c));
        }, targetKernel == null ? null : targetKernel.addRef(), sourceKernel == null ? null : sourceKernel.addRef()));
    if (null != sourceKernel)
      sourceKernel.freeRef();
    Tensor temp_05_0013 = convolutionLayer.getKernel();
    temp_05_0013.set(targetKernel == null ? null : targetKernel.addRef());
    if (null != temp_05_0013)
      temp_05_0013.freeRef();
    if (null != targetKernel)
      targetKernel.freeRef();
    AttrValue stridesArr = graphNode.getNodeDef().getAttrMap().get("strides");
    if (null != stridesArr) {
      int[] strides = stridesArr.getList().getIList().stream().mapToInt(x -> Math.toIntExact(x)).toArray();
      int strideX = strides[1];
      int strideY = strides[2];
      if (strideX > 1 || strideY > 1) {
        RefUtil.freeRef(convolutionLayer.setStrideX(strideX));
        RefUtil.freeRef(convolutionLayer.setStrideY(strideY));
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
    LRNLayer temp_05_0009 = new LRNLayer((int) width);
    LRNLayer temp_05_0014 = temp_05_0009.setAlpha(alpha * width);
    LRNLayer temp_05_0015 = temp_05_0014.setBeta(beta);
    LRNLayer temp_05_0008 = temp_05_0015.setK(bias);
    if (null != temp_05_0015)
      temp_05_0015.freeRef();
    if (null != temp_05_0014)
      temp_05_0014.freeRef();
    if (null != temp_05_0009)
      temp_05_0009.freeRef();
    return temp_05_0008;
  }

}
