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
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.tensorflow.Shape;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.core.MatMul;
import org.tensorflow.op.core.Placeholder;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.*;
import java.io.File;
import java.util.Arrays;
import java.util.Random;

import static com.simiacryptus.util.JsonUtil.toJson;

public class SimpleConvTFMnist {

  public static final String input = "image";
  public static final String weights = "fc1";
  public static final String weights_conv1 = "conv1";
  public static final String bias = "bias";
  public static final String output = "softmax";
  @Nullable
  public static final String statOutput = null;//"output/summary";

  private static byte[] getGraphDef() {
    return TensorflowUtil.makeGraph(ops -> {
      ops.withName(output).softmax(ops.add(
          ops.withName(bias).placeholder(Double.class, Placeholder.shape(Shape.make(1, 10))),
          ops.reshape(ops.transpose(
              ops.matMul(
                  ops.withName(weights).placeholder(Double.class, Placeholder.shape(Shape.make(10, 28 * 28 * 5))),
                  ops.withName("reshape_3").reshape(
                      ops.conv2D(
                          ops.withName(input).placeholder(Double.class, Placeholder.shape(Shape.make(-1, 28, 28, 1))),
                          ops.withName(weights_conv1).placeholder(Double.class,
                              Placeholder.shape(Shape.make(5, 5, 1, 5))),
                          Arrays.asList(1L, 1L, 1L, 1L), "SAME"),
                      ops.constant(new long[]{-1, 28 * 28 * 5})),
                  MatMul.transposeB(true)),
              ops.constant(new int[]{1, 0})), ops.constant(new long[]{-1, 10}))));
    });
  }

  @Nonnull
  private static RefHashMap<String, Tensor> getVariables() {
    RefHashMap<String, Tensor> variables = new RefHashMap<>();
    Tensor temp_08_0001 = new Tensor(5, 5, 1, 5);
    temp_08_0001.setByCoord(c2 -> .001 * (Math.random() - 0.5));
    RefUtil.freeRef(variables.put(weights_conv1, temp_08_0001.addRef()));
    temp_08_0001.freeRef();
    Tensor temp_08_0002 = new Tensor(10, 28 * 28 * 5);
    temp_08_0002.setByCoord(c1 -> .001 * (Math.random() - 0.5));
    RefUtil.freeRef(variables.put(weights, temp_08_0002.addRef()));
    temp_08_0002.freeRef();
    Tensor temp_08_0003 = new Tensor(10);
    temp_08_0003.setByCoord(c -> 0);
    RefUtil.freeRef(variables.put(bias, temp_08_0003.addRef()));
    temp_08_0003.freeRef();
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
        throw new RuntimeException(e);
      }
      TFLayer temp_08_0005 = new TFLayer(bytes, getVariables(), output, input);
      temp_08_0005.setSummaryOut(statOutput);
      TFLayer temp_08_0004 = temp_08_0005.addRef();
      temp_08_0005.freeRef();
      return temp_08_0004;
    });
  }

  @Nonnull
  private static GraphDef instrument(@Nonnull GraphDef graphDef) {
    return graphDef;
  }

  @Test
  public void dumpModelJson() throws Exception {
    byte[] protobufBinaryData = FileUtils.readFileToByteArray(
        new File("H:\\SimiaCryptus\\tensorflow\\tensorflow\\examples\\tutorials\\mnist\\model\\train.pb"));
    GraphModel model = new GraphModel(protobufBinaryData);
    //com.simiacryptus.ref.wrappers.RefSystem.out.println("Protobuf: " + model.graphDef);
    CharSequence json = toJson(model);
    File file = new File("model.json");
    FileUtils.write(file, json, "UTF-8");
    Desktop.getDesktop().open(file);
    RefSystem.out.println("Model: " + json);
  }

  @Test
  public void viewModelJson() throws Exception {
    TFUtil.launchTensorboard(
        new File("H:\\SimiaCryptus\\tensorflow\\tensorflow\\examples\\tutorials\\mnist\\tmp\\train\\"),
        p -> p.waitFor());
  }

  public static class MnistDemo extends MnistDemoBase {
    @Override
    protected byte[] getGraphDef() {
      return SimpleConvTFMnist.getGraphDef();
    }

    @Override
    protected Layer buildModel(@Nonnull NotebookOutput log) {
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

    public @SuppressWarnings("unused")
    void _free() { super._free(); }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    LayerTest addRef() {
      return (LayerTest) super.addRef();
    }
  }

}
