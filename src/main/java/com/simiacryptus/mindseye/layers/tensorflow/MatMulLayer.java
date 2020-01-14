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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.MatMul;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;

public class MatMulLayer extends TFLayerBase {

  private final int[] intputDims;
  private final int[] outputDims;

  public MatMulLayer(int[] intputDims, int[] outputDims) {
    super(defaultStates(intputDims, outputDims));
    this.intputDims = intputDims;
    this.outputDims = outputDims;
  }

  public MatMulLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
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
          ops.constant(RefIntStream.concat(RefIntStream.of(-1), RefArrays.stream(getOutputDims())).toArray()));
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  @Override
  public List<String> getInputNodes() {
    return Arrays.asList("input");
  }

  public int[] getIntputDims() {
    return intputDims;
  }

  public int[] getOutputDims() {
    return outputDims;
  }

  @Nonnull
  @Override
  public String getOutputNode() {
    return "output";
  }

  @Nullable
  @Override
  public String getSummaryOut() {
    return null;
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MatMulLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MatMulLayer(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MatMulLayer[] addRefs(@Nullable MatMulLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MatMulLayer::addRef).toArray((x) -> new MatMulLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MatMulLayer[][] addRefs(@Nullable MatMulLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MatMulLayer::addRefs).toArray((x) -> new MatMulLayer[x][]);
  }

  @Nonnull
  private static RefMap<String, Tensor> defaultStates(int[] intputDims, int[] outputDims) {
    RefHashMap<String, Tensor> map = new RefHashMap<>();
    int outs = Tensor.length(outputDims);
    int inputs = Tensor.length(intputDims);
    Tensor temp_17_0001 = new Tensor(outs, inputs);
    RefUtil.freeRef(map.put("weights", temp_17_0001.setByCoord(c -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    })));
    temp_17_0001.freeRef();
    return map;
  }

  @Nonnull
  public MatMulLayer set(@Nonnull final DoubleSupplier f) {
    RefMap<String, Tensor> temp_17_0003 = getWeights();
    assert temp_17_0003 != null;
    Tensor temp_17_0004 = temp_17_0003.get("weights");
    assert temp_17_0004 != null;
    RefArrays.parallelSetAll(temp_17_0004.getData(), i -> f.getAsDouble());
    temp_17_0004.freeRef();
    temp_17_0003.freeRef();
    return this.addRef();
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    assert json != null;
    json.add("inputDims", JsonUtil.toIntArray(getIntputDims()));
    json.add("outputDims", JsonUtil.toIntArray(getOutputDims()));
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MatMulLayer addRef() {
    return (MatMulLayer) super.addRef();
  }

  @Nonnull
  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    RefHashSet<String> hashSet = new RefHashSet<>();
    hashSet.add("weights");
    return hashSet;
  }

  //  @Override
  //  public boolean invertWeights() {
  //    return false;
  //  }

}
