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
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.*;

public class MaxPoolLayer extends TFLayerBase {

  private long strideX = 2L;
  private long strideY = 2L;
  private long width = 2L;
  private long height = 2L;
  private String padding = "SAME";

  public MaxPoolLayer() {
    super(defaultStates());
  }

  public MaxPoolLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    height = json.getAsJsonPrimitive("height").getAsInt();
    padding = json.getAsJsonPrimitive("padding").getAsString();
  }

  private static Map<String, Tensor> defaultStates() {
    HashMap<String, Tensor> map = new HashMap<>();
    return map;
  }

  @Nonnull
  public static MaxPoolLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxPoolLayer(json, rs);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("width", width);
    json.addProperty("height", height);
    json.addProperty("padding", padding);
    return json;
  }

  public boolean isSingleBatch() {
    return false;
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
      ops.withName(getOutputNode()).maxPool(
          ops.withName(getInputNodes().get(0)).placeholder(Double.class),
          Arrays.asList(1L, getWidth(), getHeight(), 1L),
          Arrays.asList(1L, getStrideX(), getStrideY(), 1L),
          getPadding()
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

  public long getStrideX() {
    return strideX;
  }

  public MaxPoolLayer setStrideX(long strideX) {
    this.strideX = strideX;
    return this;
  }

  public long getStrideY() {
    return strideY;
  }

  public MaxPoolLayer setStrideY(long strideY) {
    this.strideY = strideY;
    return this;
  }

  public long getWidth() {
    return width;
  }

  public MaxPoolLayer setWidth(long width) {
    this.width = width;
    return this;
  }

  public long getHeight() {
    return height;
  }

  public MaxPoolLayer setHeight(long height) {
    this.height = height;
    return this;
  }

  public String getPadding() {
    return padding;
  }

  public MaxPoolLayer setPadding(String padding) {
    this.padding = padding;
    return this;
  }
}
