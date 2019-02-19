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
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.tensorflow.NodeInstrumentation;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Shape;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.core.Placeholder;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;


public class FloatTFMnist {

  public static class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return FloatTFMnist.getGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
      log.p("This is a very simple model that performs basic logistic regression. " +
          "It is expected to be trainable to about 91% accuracy on MNIST.");
      return network(log);
    }

  }

  public static class LayerTest extends LayerTestBase {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
          {28, 28}
      };
    }

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return network();
    }

  }

  public static Layer network() {
    return network(new NullNotebookOutput());
  }

  public static final String input = "image";
  public static final String weights = "weights";
  public static final String bias = "bias";
  public static final String output = "softmax";
  public static String statOutput = "output/summary";

  public static Layer network(NotebookOutput log) {
    return log.eval(() -> {
      byte[] bytes;
      try {
        bytes = instrument(GraphDef.parseFrom(getGraphDef())).toByteArray();
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      return new TFLayer(bytes, getVariables(), output, input).setSingleBatch(true).setFloat(true).setSummaryOut(statOutput);
    });
  }

  @NotNull
  private static HashMap<String, Tensor> getVariables() {
    HashMap<String, Tensor> variables = new HashMap<>();
    variables.put(weights,
        new Tensor(1, 10, 28 * 28)
            .setByCoord(c -> .001 * (Math.random() - 0.5)));
    variables.put(bias,
        new Tensor(1, 28, 28).setByCoord(c -> 0));
    return variables;
  }

  private static GraphDef instrument(GraphDef graphDef) {
    if (null == statOutput) return graphDef;
    TensorflowUtil.validate(graphDef);
    GraphDef newDef = NodeInstrumentation.instrument(graphDef, statOutput, node -> {
      String op = node.getOp();
      if (!Arrays.asList(
          "MatMul", "BatchMatMul", "Const", "Placeholder", "Softmax", "Add"
      ).contains(op)) return null;
      NodeInstrumentation nodeInstrumentation = new NodeInstrumentation(NodeInstrumentation.getDataType(node, DataType.DT_FLOAT));
      if(node.getName().equalsIgnoreCase(input)) {
        nodeInstrumentation.setImage(28,28,1);
      }
      return nodeInstrumentation;
    });
    TensorflowUtil.validate(graphDef);
    return newDef;
  }

  private static byte[] getGraphDef() {
    byte[] bytes = TensorflowUtil.makeGraph(ops -> {
      ops.withName(output).softmax(
          ops.reshape(
              ops.batchMatMul(
                  ops.withName(weights).placeholder(
                      Float.class,
                      Placeholder.shape(Shape.make(1, 10, 28 * 28))
                  ),
                  ops.reshape(
                      ops.add(
                          ops.reshape(
                              ops.withName(bias).placeholder(
                                  Float.class,
                                  Placeholder.shape(Shape.make(1, 28, 28))
                              ),
                              ops.constant(new long[]{1, 28, 28})
                          ),
                          ops.reshape(
                              ops.withName(input).placeholder(
                                  Float.class,
                                  Placeholder.shape(Shape.make(-1, 28, 28))
                              ),
                              ops.constant(new long[]{1, 28, 28})
                          )
                      ),
                      ops.constant(new long[]{1, 28 * 28, 1})
                  )
              ),
              ops.constant(new long[]{-1, 10})
          )
      );
    });
    return bytes;
  }

}
