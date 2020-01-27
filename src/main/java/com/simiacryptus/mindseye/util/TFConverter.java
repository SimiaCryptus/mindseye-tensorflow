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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class TFConverter {

  public static RefList<TFLayer> getLayers(@Nonnull ImageNetworkPipeline pipeline) {
    return RefIntStream.range(0, pipeline.graphDefs.size()).mapToObj(i -> getLayer(pipeline, i))
        .collect(RefCollectors.toList());
  }

  @Nonnull
  public static TFLayer getLayer(@Nonnull ImageNetworkPipeline pipeline, int i) {
    GraphDef graphDef = pipeline.graphDefs.get(i);
    String output = pipeline.nodeIds().get(i);
    String input = i == 0 ? "input" : pipeline.nodeIds().get(i - 1);
    TFLayer temp_05_0004 = new TFLayer(graphDef.toByteArray(), new RefHashMap<>(), output, input);
    temp_05_0004.setFloat(true);
    TFLayer temp_05_0003 = temp_05_0004.addRef();
    temp_05_0004.freeRef();
    return temp_05_0003;
  }

  @Nonnull
  public FullyConnectedLayer getFCLayer(@Nonnull MatMulLayer matMulLayer) {
    RefMap<String, Tensor> temp_05_0010 = matMulLayer.getWeights();
    assert temp_05_0010 != null;
    Tensor weights = temp_05_0010.get("weights");
    temp_05_0010.freeRef();
    int[] intputDims = matMulLayer.getIntputDims();
    int[] outputDims = matMulLayer.getOutputDims();

    matMulLayer.freeRef();
    int[] tfView = Streams
        .concat(RefArrays.stream(outputDims),
            RefIntStream.range(0, intputDims.length).map(i -> (intputDims.length - 1) - i).map(i -> intputDims[i]))
        .toArray();
    int[] tfPermute = Streams
        .concat(RefIntStream.range(0, intputDims.length).map(i -> outputDims.length + ((intputDims.length - 1) - i)),
            RefIntStream.range(0, outputDims.length))
        .toArray();
    assert weights != null;
    Tensor temp_05_0011 = weights.reshapeCast(tfView);
    Tensor rearranged = temp_05_0011.permuteDimensions(tfPermute);

    temp_05_0011.freeRef();
    weights.freeRef();
    FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(intputDims, outputDims);
    Tensor temp_05_0012 = fullyConnectedLayer.getWeights();
    assert temp_05_0012 != null;
    temp_05_0012.set(rearranged.addRef());
    temp_05_0012.freeRef();
    rearranged.freeRef();
    return fullyConnectedLayer;
  }

  @Nonnull
  public PipelineNetwork convert(@Nonnull TFLayerBase tfLayer) {
    final PipelineNetwork converted = new PipelineNetwork(1);
    RefConcurrentHashMap<String, DAGNode> nodes = new RefConcurrentHashMap<>();
    RefUtil.freeRef(getNode(tfLayer.getOutputNode(), converted.addRef(),
        new GraphModel(tfLayer.constGraph().toByteArray()), RefUtil.addRef(nodes)));
    tfLayer.freeRef();
    nodes.freeRef();
    return converted;
  }

  @Nullable
  protected DAGNode getNode(@Nonnull String id, @Nonnull PipelineNetwork network, @Nonnull GraphModel tfModel,
                            @Nonnull RefMap<String, DAGNode> map) {
    try {
      if (!map.containsKey(id)) {
        DAGNode result = getDagNode(id, network, tfModel, map.addRef());
        if (!map.containsKey(id)) {
          RefUtil.freeRef(map.put(id, result));
        } else {
          result.freeRef();
        }
      } else {
        network.freeRef();
      }
      DAGNode node = map.get(id);
      map.freeRef();
      return node;
    } catch (Throwable e) {
      throw new RuntimeException("Error converting " + id, e);
    }
  }

  @NotNull
  private DAGNode getDagNode(@Nonnull String id, @Nonnull PipelineNetwork network, @Nonnull GraphModel tfModel, @Nonnull RefMap<String, DAGNode> map) {
    GraphModel.GraphNode graphNode = tfModel.getChild(id);
    assert null != graphNode;
    try {
      if (graphNode.getOp().equals("Conv2D")) {
        return network.add(getConv2D(graphNode), getNode(graphNode.getInputKeys().get(0),
              network.addRef(), tfModel, map));
      } else if (graphNode.getOp().equals("BiasAdd")) {
        return network.add(getBiasAdd(graphNode), getNode(graphNode.getInputKeys().get(0),
              network.addRef(), tfModel, map));
      } else if (graphNode.getOp().equals("Relu")) {
        return network.add(new ActivationLayer(ActivationLayer.Mode.RELU), getNode(graphNode.getInputKeys().get(0),
              network.addRef(), tfModel, map));
      } else if (graphNode.getOp().equals("LRN")) {
        return network.add(getLRNLayer(graphNode), getNode(graphNode.getInputKeys().get(0),
              network.addRef(), tfModel, map));
      } else if (graphNode.getOp().equals("MaxPool")) {
        return network.add(getPoolingLayer(graphNode), getNode(graphNode.getInputKeys().get(0),
              network.addRef(), tfModel, map));
      } else if (graphNode.getOp().equals("Concat")) {
        List<String> inputKeys = graphNode.getInputKeys();
        InnerNode innerNode = network.add(new ImgConcatLayer(),
            inputKeys.stream().skip(1)
                .map(inputKey -> getNode(inputKey,
                    network.addRef(), tfModel, RefUtil.addRef(map)))
                .toArray(i -> new DAGNode[i]));
        map.freeRef();
        return innerNode;
      } else if (graphNode.getOp().equals("Placeholder")) {
        map.freeRef();
        return network.getInput(0);
      } else {
        map.freeRef();
        throw new IllegalArgumentException(graphNode.getOp());
      }
    } finally {
      network.freeRef();
    }
  }

  @Nonnull
  protected PoolingLayer getPoolingLayer(@Nonnull GraphModel.GraphNode graphNode) {
    PoolingLayer temp_05_0005 = new PoolingLayer();
    temp_05_0005.setMode(PoolingLayer.PoolingMode.Max);
    PoolingLayer poolingLayer = temp_05_0005.addRef();
    temp_05_0005.freeRef();
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

  @Nonnull
  protected ImgBandBiasLayer getBiasAdd(@Nonnull GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    double[] data = dataNode.getData();
    assert data != null;
    Tensor tensor = new Tensor(data, data.length);
    ImgBandBiasLayer temp_05_0006 = new ImgBandBiasLayer(data.length);
    temp_05_0006.set(tensor.addRef());
    ImgBandBiasLayer temp_05_0001 = temp_05_0006.addRef();
    temp_05_0006.freeRef();
    tensor.freeRef();
    return temp_05_0001;
  }

  @Nonnull
  protected Layer getConv2D(@Nonnull GraphModel.GraphNode graphNode) {
    GraphModel.GraphNode dataNode = graphNode.getInputs().get(1);
    assert dataNode.getOp().equals("Const");
    int[] kernelDims = RefArrays.stream(dataNode.getShape()).mapToInt(x -> (int) x).toArray();
    double[] data = dataNode.getData();
    if (kernelDims.length == 0) {
      assert data != null;
      kernelDims = new int[]{data.length};
    }
    Tensor temp_05_0007 = new Tensor(data, kernelDims[3], kernelDims[2], kernelDims[1], kernelDims[0]);
    Tensor sourceKernel = temp_05_0007.invertDimensions();
    temp_05_0007.freeRef();
    int[] sourceKernelDimensions = sourceKernel.getDimensions();
    //    ConvolutionLayer convolutionLayer = new ConvolutionLayer(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2], sourceKernelDimensions[3]);
    SimpleConvolutionLayer convolutionLayer = new SimpleConvolutionLayer(sourceKernelDimensions[0],
        sourceKernelDimensions[1], sourceKernelDimensions[2] * sourceKernelDimensions[3]);
    Tensor targetKernel = new Tensor(sourceKernelDimensions[0], sourceKernelDimensions[1], sourceKernelDimensions[2],
        sourceKernelDimensions[3]);
    sourceKernel.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
      int[] coord = c.getCoords();
      targetKernel.set((sourceKernelDimensions[0] - 1) - coord[0], (sourceKernelDimensions[1] - 1) - coord[1], coord[2],
          coord[3], sourceKernel.get(c));
    }, targetKernel.addRef(), sourceKernel.addRef()));
    sourceKernel.freeRef();
    Tensor temp_05_0013 = convolutionLayer.getKernel();
    assert temp_05_0013 != null;
    temp_05_0013.set(targetKernel.addRef());
    temp_05_0013.freeRef();
    targetKernel.freeRef();
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

  @Nonnull
  private LRNLayer getLRNLayer(@Nonnull GraphModel.GraphNode graphNode) {
    Map<String, AttrValue> attrMap = graphNode.getNodeDef().getAttrMap();
    long depth_radius = attrMap.get("depth_radius").getI();
    float alpha = attrMap.get("alpha").getF();
    float bias = attrMap.get("bias").getF();
    float beta = attrMap.get("beta").getF();
    long width = depth_radius * 2 + 1;
    LRNLayer temp_05_0009 = new LRNLayer((int) width);
    temp_05_0009.setAlpha(alpha * width);
    LRNLayer temp_05_0014 = temp_05_0009.addRef();
    temp_05_0014.setBeta(beta);
    LRNLayer temp_05_0015 = temp_05_0014.addRef();
    temp_05_0015.setK(bias);
    LRNLayer temp_05_0008 = temp_05_0015.addRef();
    temp_05_0015.freeRef();
    temp_05_0014.freeRef();
    temp_05_0009.freeRef();
    return temp_05_0008;
  }

}
