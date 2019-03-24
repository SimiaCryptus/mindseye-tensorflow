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
import java.util.*;

public class SummaryLayer extends TFLayerBase {

  private String tag;

  public SummaryLayer(String name) {
    super(new HashMap<>());
    this.setTag(name);
  }

  public SummaryLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    tag = json.get("tag").getAsString();
  }

  @Nonnull
  public static SummaryLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SummaryLayer(json, rs);
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("tag", tag);
    return json;
  }

  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    HashSet<String> hashSet = new HashSet<>();
    return hashSet;
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).identity(
          ops.withName(getInputNodes().get(0)).placeholder(Double.class)
      );
      return NodeInstrumentation.instrument(
          GraphDef.parseFrom(graph.toGraphDef()),
          getSummaryOut(),
          node -> node.getName().equals(getInputNodes().get(0)) ? new NodeInstrumentation(NodeInstrumentation.getDataType(node, DataType.DT_DOUBLE)) : null
      );
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public String getSummaryOut() {
    return "summary";
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  @Override
  public List<String> getInputNodes() {
    return Arrays.asList(tag);
  }

  public String getTag() {
    return tag;
  }

  public SummaryLayer setTag(String tag) {
    this.tag = tag;
    return this;
  }
}
