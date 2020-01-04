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
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;

public @com.simiacryptus.ref.lang.RefAware
class BiasLayer extends TFLayerBase {

  public BiasLayer(int... intputDims) {
    super(defaultStates(intputDims));
  }

  public BiasLayer(JsonObject json, com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).add(ops.withName("bias").placeholder(Double.class),
          ops.withName(getInputNodes().get(0)).placeholder(Double.class));
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<String> getInputNodes() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList("input");
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
  public static BiasLayer fromJson(@Nonnull final JsonObject json,
                                   com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new BiasLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  BiasLayer[] addRefs(BiasLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BiasLayer::addRef)
        .toArray((x) -> new BiasLayer[x]);
  }

  public static @SuppressWarnings("unused")
  BiasLayer[][] addRefs(BiasLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BiasLayer::addRefs)
        .toArray((x) -> new BiasLayer[x][]);
  }

  private static com.simiacryptus.ref.wrappers.RefMap<String, Tensor> defaultStates(int[] intputDims) {
    com.simiacryptus.ref.wrappers.RefHashMap<String, Tensor> map = new com.simiacryptus.ref.wrappers.RefHashMap<>();
    map.put("bias", new Tensor(intputDims).setByCoord(c -> {
      return 0;
    }));
    return map;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  BiasLayer addRef() {
    return (BiasLayer) super.addRef();
  }

  @Override
  protected com.simiacryptus.ref.wrappers.RefSet<String> getDataKeys(JsonObject json) {
    com.simiacryptus.ref.wrappers.RefHashSet<String> hashSet = new com.simiacryptus.ref.wrappers.RefHashSet<>();
    hashSet.add("bias");
    return hashSet;
  }

}
