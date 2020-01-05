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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

public @RefAware
class SoftmaxLayer extends TFLayerBase {

  public SoftmaxLayer() {
    super(new RefHashMap<>());
  }

  public SoftmaxLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      RefList<String> temp_24_0001 = getInputNodes();
      ops.withName(getOutputNode()).softmax(ops.withName(temp_24_0001.get(0)).placeholder(Double.class));
      if (null != temp_24_0001)
        temp_24_0001.freeRef();
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
  public static SoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  SoftmaxLayer[] addRefs(SoftmaxLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayer::addRef)
        .toArray((x) -> new SoftmaxLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SoftmaxLayer[][] addRefs(SoftmaxLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayer::addRefs)
        .toArray((x) -> new SoftmaxLayer[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SoftmaxLayer addRef() {
    return (SoftmaxLayer) super.addRef();
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    return new RefHashSet<>();
  }

}
