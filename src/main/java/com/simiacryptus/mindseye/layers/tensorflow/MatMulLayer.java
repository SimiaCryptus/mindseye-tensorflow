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

package com.simiacryptus.mindseye.layers.tensorflow;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.Util;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.MatMul;

import javax.annotation.Nonnull;
import java.util.*;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

public class MatMulLayer extends TFLayerBase {

  private final int[] intputDims;
  private final int[] outputDims;

  public boolean isSingleBatch() {
    return false;
  }

  public MatMulLayer(int[] intputDims, int[] outputDims) {
    super(defaultStates(intputDims, outputDims));
    this.intputDims = intputDims;
    this.outputDims = outputDims;
  }

  private static Map<String, Tensor> defaultStates(int[] intputDims, int[] outputDims) {
    HashMap<String, Tensor> map = new HashMap<>();
    int outs = Tensor.length(outputDims);
    int inputs = Tensor.length(intputDims);
    map.put("weights", new Tensor(outs, inputs).setByCoord(c->{
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      final double v = (1 - 2 * fate) * ratio;
      return v;
    }));
    return map;
  }

  @Nonnull
  public MatMulLayer set(@Nonnull final DoubleSupplier f) {
    Arrays.parallelSetAll(getWeights().get("weights").getData(), i -> f.getAsDouble());
    return this;
  }

  @Nonnull
  public static MatMulLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MatMulLayer(json, rs);
  }

  public MatMulLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    intputDims = toIntArray(json.get("inputDims").getAsJsonArray());
    outputDims = toIntArray(json.get("outputDims").getAsJsonArray());
  }

  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    HashSet<String> hashSet = new HashSet<>();
    hashSet.add("weights");
    return hashSet;
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.add("inputDims", toIntArray(intputDims));
    json.add("outputDims", toIntArray(outputDims));
    return json;
  }

  public static int[] toIntArray(JsonArray array) {
    int[] ints = new int[array.size()];
    for (int i = 0; i < ints.length; i++) {
      ints[i] = array.get(i).getAsInt();
    }
    return ints;
  }

  public static JsonArray toIntArray(int[] array) {
    JsonArray jsonElements = new JsonArray();
    Arrays.stream(array).forEach(jsonElements::add);
    return jsonElements;
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).reshape(
          ops.transpose(
              ops.matMul(
                  ops.withName("weights").placeholder(Double.class),
                  ops.reshape(
                      ops.withName(getInputNodes().get(0)).placeholder(Double.class),
                      ops.constant(new long[]{-1, Tensor.length(intputDims)})
                  ),
                  MatMul.transposeB(true)
              ),
              ops.constant(new int[]{1,0})
          ),
          ops.constant(IntStream.concat(
              IntStream.of(-1),
              Arrays.stream(outputDims)
          ).toArray())
      );
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public String getSummaryOut() {
    return null;
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  @Override
  public List<String> getInputNodes() {
    return Arrays.asList("input");
  }

}
