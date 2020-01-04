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

import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.MatMul;

import javax.annotation.Nonnull;
import java.util.function.DoubleSupplier;

public @com.simiacryptus.ref.lang.RefAware
class MatMulLayer extends TFLayerBase {

  private final int[] intputDims;
  private final int[] outputDims;

  public MatMulLayer(int[] intputDims, int[] outputDims) {
    super(defaultStates(intputDims, outputDims));
    this.intputDims = intputDims;
    this.outputDims = outputDims;
  }

  public MatMulLayer(JsonObject json, com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
    intputDims = JsonUtil.toIntArray(json.get("inputDims").getAsJsonArray());
    outputDims = JsonUtil.toIntArray(json.get("outputDims").getAsJsonArray());
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).reshape(
          ops.transpose(
              ops.matMul(ops.withName("weights").placeholder(Double.class),
                  ops.reshape(ops.withName(getInputNodes().get(0)).placeholder(Double.class),
                      ops.constant(new long[]{-1, Tensor.length(getIntputDims())})),
                  MatMul.transposeB(true)),
              ops.constant(new int[]{1, 0})),
          ops.constant(
              com.simiacryptus.ref.wrappers.RefIntStream.concat(com.simiacryptus.ref.wrappers.RefIntStream.of(-1),
                  com.simiacryptus.ref.wrappers.RefArrays.stream(getOutputDims())).toArray()));
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<String> getInputNodes() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList("input");
  }

  public int[] getIntputDims() {
    return intputDims;
  }

  public int[] getOutputDims() {
    return outputDims;
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  @Override
  public String getSummaryOut() {
    return null;
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MatMulLayer fromJson(@Nonnull final JsonObject json,
                                     com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new MatMulLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  MatMulLayer[] addRefs(MatMulLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MatMulLayer::addRef)
        .toArray((x) -> new MatMulLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MatMulLayer[][] addRefs(MatMulLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MatMulLayer::addRefs)
        .toArray((x) -> new MatMulLayer[x][]);
  }

  private static com.simiacryptus.ref.wrappers.RefMap<String, Tensor> defaultStates(int[] intputDims,
                                                                                    int[] outputDims) {
    com.simiacryptus.ref.wrappers.RefHashMap<String, Tensor> map = new com.simiacryptus.ref.wrappers.RefHashMap<>();
    int outs = Tensor.length(outputDims);
    int inputs = Tensor.length(intputDims);
    map.put("weights", new Tensor(outs, inputs).setByCoord(c -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    }));
    return map;
  }

  @Nonnull
  public MatMulLayer set(@Nonnull final DoubleSupplier f) {
    com.simiacryptus.ref.wrappers.RefArrays.parallelSetAll(getWeights().get("weights").getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.add("inputDims", JsonUtil.toIntArray(getIntputDims()));
    json.add("outputDims", JsonUtil.toIntArray(getOutputDims()));
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MatMulLayer addRef() {
    return (MatMulLayer) super.addRef();
  }

  @Override
  protected com.simiacryptus.ref.wrappers.RefSet<String> getDataKeys(JsonObject json) {
    com.simiacryptus.ref.wrappers.RefHashSet<String> hashSet = new com.simiacryptus.ref.wrappers.RefHashSet<>();
    hashSet.add("weights");
    return hashSet;
  }

  //  @Override
  //  public boolean invertWeights() {
  //    return false;
  //  }

}
