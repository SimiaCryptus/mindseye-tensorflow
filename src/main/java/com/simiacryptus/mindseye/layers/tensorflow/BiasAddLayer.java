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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

public @RefAware
class BiasAddLayer extends TFLayerBase {

  public BiasAddLayer(int... intputDims) {
    super(defaultStates(intputDims));
  }

  public BiasAddLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      RefList<String> temp_10_0002 = getInputNodes();
      ops.withName(getOutputNode()).biasAdd(ops.withName(temp_10_0002.get(0)).placeholder(Double.class),
          ops.withName("bias").placeholder(Double.class));
      if (null != temp_10_0002)
        temp_10_0002.freeRef();
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public RefList<String> getInputNodes() {
    return RefArrays.asList("input");
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
  public static BiasAddLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasAddLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  BiasAddLayer[] addRefs(BiasAddLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasAddLayer::addRef)
        .toArray((x) -> new BiasAddLayer[x]);
  }

  public static @SuppressWarnings("unused")
  BiasAddLayer[][] addRefs(BiasAddLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasAddLayer::addRefs)
        .toArray((x) -> new BiasAddLayer[x][]);
  }

  private static RefMap<String, Tensor> defaultStates(int[] intputDims) {
    RefHashMap<String, Tensor> map = new RefHashMap<>();
    Tensor temp_10_0001 = new Tensor(intputDims);
    RefUtil.freeRef(map.put("bias", temp_10_0001.setByCoord(c -> {
      return 0;
    })));
    if (null != temp_10_0001)
      temp_10_0001.freeRef();
    return map;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  BiasAddLayer addRef() {
    return (BiasAddLayer) super.addRef();
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    RefHashSet<String> hashSet = new RefHashSet<>();
    hashSet.add("bias");
    return hashSet;
  }

}
