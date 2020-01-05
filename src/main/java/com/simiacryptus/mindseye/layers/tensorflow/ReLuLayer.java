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
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

public @RefAware
class ReLuLayer extends TFLayerBase {

  public ReLuLayer() {
    super(defaultStates());
  }

  public ReLuLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).relu(ops.withName(getInputNodes().get(0)).placeholder(Double.class));
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
  public static ReLuLayer fromJson(@Nonnull final JsonObject json,
                                   Map<CharSequence, byte[]> rs) {
    return new ReLuLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  ReLuLayer[] addRefs(ReLuLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuLayer::addRef)
        .toArray((x) -> new ReLuLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ReLuLayer[][] addRefs(ReLuLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuLayer::addRefs)
        .toArray((x) -> new ReLuLayer[x][]);
  }

  private static RefMap<String, Tensor> defaultStates() {
    return new RefHashMap<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ReLuLayer addRef() {
    return (ReLuLayer) super.addRef();
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    return new RefHashSet<>();
  }

}
