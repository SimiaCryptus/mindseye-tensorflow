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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.tensorflow.NodeInstrumentation;
import com.simiacryptus.tensorflow.TensorflowUtil;
import com.simiacryptus.util.Util;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.linalg.MatMul;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public class ConvTFMnist {

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
      Operand<Double> conv1 = ops.math.add(
          ops.withName(bias1).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1, 1, bands1))),
          ops.nn.relu(ops.nn.maxPool(
              ops.withName("conv2d_0").nn.conv2d(
                  ops.withName(input).placeholder(Double.class, Placeholder.shape(Shape.make(-1, 28, 28, 1))),
                  ops.withName(ConvTFMnist.conv1).placeholder(Double.class, Placeholder.shape(Shape.make(5, 5, 1, bands1))),
                  Arrays.asList(1L, 1L, 1L, 1L),
                  "SAME"),
              ops.constant(new int[]{1, 2, 2, 1}),
              ops.constant(new int[]{1, 2, 2, 1}),
              "SAME")));
      int bands2 = 64;
      Operand<Double> conv2 = ops.math.add(
          ops.withName(bias2).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1, 1, bands2))),
          ops.nn.relu(ops.nn.maxPool(
              ops.withName("conv2d_1").nn.conv2d(conv1,
                  ops.withName(ConvTFMnist.conv2).placeholder(Double.class,
                      Placeholder.shape(Shape.make(5, 5, bands1, bands2))),
                  Arrays.asList(1L, 1L, 1L, 1L),
                  "SAME"),
              ops.constant(new int[]{1, 2, 2, 1}),
              ops.constant(new int[]{1, 2, 2, 1}),
              "SAME")));
      Operand<Double> fc1 = ops.math.add(
          ops.withName(bias3).placeholder(Double.class, Placeholder.shape(Shape.make(1, 1024))),
          ops.nn.relu(ops.linalg.transpose(
              ops.linalg.matMul(
                  ops.withName(ConvTFMnist.fc1).placeholder(Double.class,
                      Placeholder.shape(Shape.make(1024, 7 * 7 * bands2))),
                  ops.reshape(conv2, ops.constant(new long[]{-1, 7 * 7 * bands2})), MatMul.transposeB(true)),
              ops.constant(new int[]{1, 0}))));
      ops.withName(output).reshape(fc1, ops.constant(new long[]{-1, 1024}));
    });
  }

  @Nonnull
  private static RefHashMap<String, Tensor> getVariables() {
    RefHashMap<String, Tensor> variables = new RefHashMap<>();
    Tensor temp_16_0001 = new Tensor(5, 5, 1, 32);
    temp_16_0001.randomize(.001);
    RefUtil.freeRef(variables.put(conv1, temp_16_0001.addRef()));
    temp_16_0001.freeRef();
    Tensor temp_16_0002 = new Tensor(5, 5, 32, 64);
    temp_16_0002.randomize(.001);
    RefUtil.freeRef(variables.put(conv2, temp_16_0002.addRef()));
    temp_16_0002.freeRef();
    RefUtil.freeRef(variables.put(bias1, new Tensor(1, 1, 1, 32)));
    RefUtil.freeRef(variables.put(bias2, new Tensor(1, 1, 1, 64)));
    RefUtil.freeRef(variables.put(bias3, new Tensor(1, 1024)));
    Tensor temp_16_0003 = new Tensor(1024, 7 * 7 * 64);
    temp_16_0003.randomize(.001);
    RefUtil.freeRef(variables.put(fc1, temp_16_0003.addRef()));
    temp_16_0003.freeRef();
    return variables;
  }

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static Layer network(@Nonnull NotebookOutput log) {
    return log.eval(() -> {
      byte[] bytes;
      try {
        bytes = instrument(GraphDef.parseFrom(getGraphDef())).toByteArray();
      } catch (InvalidProtocolBufferException e) {
        throw Util.throwException(e);
      }
      TFLayer temp_16_0005 = new TFLayer(bytes, getVariables(), output, input);
      temp_16_0005.setSummaryOut(statOutput);
      Layer temp_16_0004 = stochasticClassificationLayer(temp_16_0005.addRef(), Math.pow(0.5, 1.0), 5,
          0.001);
      temp_16_0005.freeRef();
      return temp_16_0004;
    });
  }

  @Nonnull
  public static Layer stochasticClassificationLayer(@Nullable Layer inner, double density, int samples, double initialWeight) {
    PipelineNetwork stochasticTerminal = new PipelineNetwork(1);
    RefUtil.freeRef(stochasticTerminal.add(BinaryNoiseLayer.maskLayer(density)));
    FullyConnectedLayer temp_16_0006 = new FullyConnectedLayer(new int[]{1024}, new int[]{10});
    temp_16_0006.randomize(initialWeight);
    RefUtil.freeRef(stochasticTerminal.add(temp_16_0006.addRef()));
    temp_16_0006.freeRef();
    RefUtil.freeRef(stochasticTerminal.add(new BiasLayer(10)));
    RefUtil.freeRef(stochasticTerminal.add(new SoftmaxLayer()));
    PipelineNetwork pipeline = new PipelineNetwork(1);
    RefUtil.freeRef(pipeline.add(inner == null ? null : inner.addRef()));
    if (null != inner)
      inner.freeRef();
    RefUtil.freeRef(pipeline.add(
        new StochasticSamplingSubnetLayer(stochasticTerminal.addRef(), samples)));
    stochasticTerminal.freeRef();
    return pipeline;
  }

  @Nonnull
  private static GraphDef instrument(@Nonnull GraphDef graphDef) {
    TensorflowUtil.validate(graphDef);
    GraphDef newDef = NodeInstrumentation.instrument(graphDef, statOutput, node -> {
      String op = node.getOp();
      if (!Arrays.asList(
          "MatMul",
          "BatchMatMul",
          "Const",
          "Placeholder",
          "Softmax",
          "Add",
          "Conv2D"
      ).contains(op)) return null;
      //      if (node.getName().equalsIgnoreCase(input)) {
      //        nodeInstrumentation.setImage(28, 28, 1);
      //      }
      return new NodeInstrumentation(NodeInstrumentation.getDataType(node, DataType.DT_DOUBLE));
    });
    TensorflowUtil.validate(graphDef);
    return newDef;
  }

  public static class MnistDemo extends MnistDemoBase {
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

  public static class LayerTest extends LayerTestBase {

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
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

  }

}
