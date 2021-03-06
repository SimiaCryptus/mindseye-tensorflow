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
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.tensorflow.TensorflowUtil;
import com.simiacryptus.util.Util;
import org.tensorflow.Shape;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.linalg.MatMul;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class FloatTFMnist {

  public static final String input = "image";
  public static final String weights = "fc1";
  public static final String bias = "bias";
  public static final String output = "softmax";
  @Nullable
  public static final String statOutput = null; //"output/summary";

  private static byte[] getGraphDef() {
    return TensorflowUtil.makeGraph(ops -> {
      ops.withName(output)
          .nn.softmax(
          ops.reshape(
              ops.linalg.transpose(
                  ops.linalg.matMul(ops.withName(weights).placeholder(Float.class, Placeholder
                          .shape(Shape.make(10, 28 * 28))), ops
                          .reshape(
                              ops.math.add(
                                  ops.reshape(
                                      ops.withName(bias).placeholder(Float.class,
                                          Placeholder.shape(Shape.make(1, 28, 28))),
                                      ops.constant(new long[]{1, 28, 28})),
                                  ops.reshape(
                                      ops.withName(input).placeholder(Float.class,
                                          Placeholder.shape(Shape.make(-1, 28, 28))),
                                      ops.constant(new long[]{-1, 28, 28}))),
                              ops.constant(new long[]{-1, 28 * 28})),
                      MatMul.transposeB(true)),
                  ops.constant(new int[]{1, 0})),
              ops.constant(new long[]{-1, 10})));
    });
  }

  @Nonnull
  private static RefHashMap<String, Tensor> getVariables() {
    RefHashMap<String, Tensor> variables = new RefHashMap<>();
    Tensor temp_13_0001 = new Tensor(10, 28 * 28);
    temp_13_0001.setByCoord(c1 -> .001 * (Math.random() - 0.5));
    RefUtil.freeRef(variables.put(weights, temp_13_0001.addRef()));
    temp_13_0001.freeRef();
    Tensor temp_13_0002 = new Tensor(1, 28, 28);
    temp_13_0002.setByCoord(c -> 0);
    RefUtil.freeRef(variables.put(bias, temp_13_0002.addRef()));
    temp_13_0002.freeRef();
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
      TFLayer temp_13_0004 = new TFLayer(bytes, getVariables(), output, input);
      temp_13_0004.setFloat(true);
      TFLayer temp_13_0005 = temp_13_0004.addRef();
      temp_13_0005.setSummaryOut(statOutput);
      TFLayer temp_13_0003 = temp_13_0005.addRef();
      temp_13_0005.freeRef();
      temp_13_0004.freeRef();
      return temp_13_0003;
    });
  }

  @Nonnull
  private static GraphDef instrument(@Nonnull GraphDef graphDef) {
    return graphDef;
  }

  public static class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return FloatTFMnist.getGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
      log.p("This is a very simple model that performs basic logistic regression. "
          + "It is expected to be trainable to about 91% accuracy on MNIST.");
      return network(log);
    }
  }

  public static class LayerTest extends LayerTestBase {

    @Nonnull
    @Override
    public Layer getLayer() {
      return network();
    }

    @Nullable
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }

    @Nonnull
    @Override
    public int[][] getSmallDims() {
      return new int[][]{{28, 28}};
    }

  }

}
