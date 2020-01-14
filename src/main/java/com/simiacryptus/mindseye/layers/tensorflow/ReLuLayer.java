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
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefHashSet;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefSet;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ReLuLayer extends TFLayerBase {

  public ReLuLayer() {
    super(defaultStates());
  }

  public ReLuLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
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

  @Nonnull
  @Override
  public List<String> getInputNodes() {
    return Arrays.asList("input");
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
  public static ReLuLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ReLuLayer(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ReLuLayer[] addRefs(@Nullable ReLuLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuLayer::addRef).toArray((x) -> new ReLuLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ReLuLayer[][] addRefs(@Nullable ReLuLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ReLuLayer::addRefs).toArray((x) -> new ReLuLayer[x][]);
  }

  @Nonnull
  private static RefMap<String, Tensor> defaultStates() {
    return new RefHashMap<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ReLuLayer addRef() {
    return (ReLuLayer) super.addRef();
  }

  @Nonnull
  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    return new RefHashSet<>();
  }

}
