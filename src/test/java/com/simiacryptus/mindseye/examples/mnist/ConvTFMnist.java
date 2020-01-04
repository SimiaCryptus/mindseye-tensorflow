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

package com.simiacryptus.mindseye.examples.mnist;

import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.tensorflow.NodeInstrumentation;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.core.MatMul;
import org.tensorflow.op.core.Placeholder;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public @com.simiacryptus.ref.lang.RefAware
class ConvTFMnist {

  public static final String input = "image";
  public static final String fc1 = "fc1";
  public static final String conv1 = "conv1";
  public static final String bias1 = "bias1";
  public static final String conv2 = "conv2";
  public static final String bias2 = "bias2";
  public static final String bias3 = "bias3";
  public static final String output = "softmax";
  public static final String statOutput = "output/summary";

  private static byte[] getGraphDef() {
    return TensorflowUtil.makeGraph(ops -> {
      int bands1 = 32;
      Operand<Double> conv1 = ops.add(
          ops.withName(bias1).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1, 1, bands1))),
          ops.relu(ops.maxPool(
              ops.withName("conv2d_0").conv2D(
                  ops.withName(input).placeholder(Double.class, Placeholder.shape(Shape.make(-1, 28, 28, 1))),
                  ops.withName(ConvTFMnist.conv1).placeholder(Double.class,
                      Placeholder.shape(Shape.make(5, 5, 1, bands1))),
                  com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 1L, 1L, 1L), "SAME"),
              com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 2L, 2L, 1L),
              com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 2L, 2L, 1L), "SAME")));
      int bands2 = 64;
      Operand<Double> conv2 = ops.add(
          ops.withName(bias2).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1, 1, bands2))),
          ops.relu(ops.maxPool(
              ops.withName("conv2d_1").conv2D(conv1,
                  ops.withName(ConvTFMnist.conv2).placeholder(Double.class,
                      Placeholder.shape(Shape.make(5, 5, bands1, bands2))),
                  com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 1L, 1L, 1L), "SAME"),
              com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 2L, 2L, 1L),
              com.simiacryptus.ref.wrappers.RefArrays.asList(1L, 2L, 2L, 1L), "SAME")));
      Operand<Double> fc1 = ops.add(
          ops.withName(bias3).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1024))),
          ops.relu(ops.transpose(
              ops.matMul(
                  ops.withName(ConvTFMnist.fc1).placeholder(Double.class,
                      Placeholder.shape(Shape.make(1024, 7 * 7 * bands2))),
                  ops.reshape(conv2, ops.constant(new long[]{-1, 7 * 7 * bands2})), MatMul.transposeB(true)),
              ops.constant(new int[]{1, 0}))));
      ops.withName(output).reshape(fc1, ops.constant(new long[]{-1, 1024}));
    });
  }

  @NotNull
  private static com.simiacryptus.ref.wrappers.RefHashMap<String, Tensor> getVariables() {
    com.simiacryptus.ref.wrappers.RefHashMap<String, Tensor> variables = new com.simiacryptus.ref.wrappers.RefHashMap<>();
    variables.put(conv1, new Tensor(5, 5, 1, 32).randomize(.001));
    variables.put(conv2, new Tensor(5, 5, 32, 64).randomize(.001));
    variables.put(bias1, new Tensor(1, 1, 1, 32));
    variables.put(bias2, new Tensor(1, 1, 1, 64));
    variables.put(bias3, new Tensor(1, 1024));
    variables.put(fc1, new Tensor(1024, 7 * 7 * 64).randomize(.001));
    return variables;
  }

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      byte[] bytes;
      try {
        bytes = instrument(GraphDef.parseFrom(getGraphDef())).toByteArray();
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      return stochasticClassificationLayer(new TFLayer(bytes, getVariables(), output, input).setSummaryOut(statOutput),
          Math.pow(0.5, 1.0), 5, 0.001);
    });
  }

  @NotNull
  public static Layer stochasticClassificationLayer(Layer inner, double density, int samples, double initialWeight) {
    PipelineNetwork stochasticTerminal = new PipelineNetwork(1);
    stochasticTerminal.add(BinaryNoiseLayer.maskLayer(density));
    stochasticTerminal.add(new FullyConnectedLayer(new int[]{1024}, new int[]{10}).randomize(initialWeight));
    stochasticTerminal.add(new BiasLayer(10));
    stochasticTerminal.add(new SoftmaxLayer());
    PipelineNetwork pipeline = new PipelineNetwork(1);
    pipeline.add(inner);
    pipeline.add(new StochasticSamplingSubnetLayer(stochasticTerminal, samples));
    return pipeline;
  }

  private static GraphDef instrument(GraphDef graphDef) {
    if (null == statOutput)
      return graphDef;
    TensorflowUtil.validate(graphDef);
    GraphDef newDef = NodeInstrumentation.instrument(graphDef, statOutput, node -> {
      String op = node.getOp();
      if (!com.simiacryptus.ref.wrappers.RefArrays
          .asList("MatMul", "BatchMatMul", "Const", "Placeholder", "Softmax", "Add", "Conv2D").contains(op))
        return null;
      //      if (node.getName().equalsIgnoreCase(input)) {
      //        nodeInstrumentation.setImage(28, 28, 1);
      //      }
      return new NodeInstrumentation(NodeInstrumentation.getDataType(node, DataType.DT_DOUBLE));
    });
    TensorflowUtil.validate(graphDef);
    return newDef;
  }

  public static @com.simiacryptus.ref.lang.RefAware
  class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return ConvTFMnist.getGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
      timeout = 15 * 60;
      log.p("This is a very simple model that performs basic logistic regression. "
          + "It is expected to be trainable to about 91% accuracy on MNIST.");
      return network(log);
    }

  }

  public static @com.simiacryptus.ref.lang.RefAware
  class LayerTest extends LayerTestBase {

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    public static @SuppressWarnings("unused")
    LayerTest[] addRefs(LayerTest[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(LayerTest::addRef)
          .toArray((x) -> new LayerTest[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{28, 28}};
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return network();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    LayerTest addRef() {
      return (LayerTest) super.addRef();
    }

  }

}
