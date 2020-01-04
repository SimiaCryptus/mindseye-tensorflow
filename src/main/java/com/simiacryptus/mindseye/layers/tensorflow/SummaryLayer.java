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
import com.simiacryptus.tensorflow.NodeInstrumentation;
import org.tensorflow.Graph;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;

public @com.simiacryptus.ref.lang.RefAware
class SummaryLayer extends TFLayerBase {

  private String tag;

  public SummaryLayer(String name) {
    super(new com.simiacryptus.ref.wrappers.RefHashMap<>());
    this.setTag(name);
  }

  public SummaryLayer(JsonObject json, com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
    tag = json.get("tag").getAsString();
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).identity(ops.withName(getInputNodes().get(0)).placeholder(Double.class));
      return NodeInstrumentation.instrument(GraphDef.parseFrom(graph.toGraphDef()), getSummaryOut(),
          node -> node.getName().equals(getInputNodes().get(0))
              ? new NodeInstrumentation(NodeInstrumentation.getDataType(node, DataType.DT_DOUBLE))
              : null);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<String> getInputNodes() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(tag);
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  @Override
  public String getSummaryOut() {
    return "summary";
  }

  public String getTag() {
    return tag;
  }

  public void setTag(String tag) {
    this.tag = tag;
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SummaryLayer fromJson(@Nonnull final JsonObject json,
                                      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SummaryLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  SummaryLayer[] addRefs(SummaryLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SummaryLayer::addRef)
        .toArray((x) -> new SummaryLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SummaryLayer[][] addRefs(SummaryLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SummaryLayer::addRefs)
        .toArray((x) -> new SummaryLayer[x][]);
  }

  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("tag", tag);
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SummaryLayer addRef() {
    return (SummaryLayer) super.addRef();
  }

  @Override
  protected com.simiacryptus.ref.wrappers.RefSet<String> getDataKeys(JsonObject json) {
    return new com.simiacryptus.ref.wrappers.RefHashSet<>();
  }
}
