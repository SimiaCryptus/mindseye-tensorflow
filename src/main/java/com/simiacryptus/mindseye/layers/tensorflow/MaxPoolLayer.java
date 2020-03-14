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
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefMap;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
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

  public MaxPoolLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    height = json.getAsJsonPrimitive("height").getAsInt();
    padding = json.getAsJsonPrimitive("padding").getAsString();
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).nn.maxPool(
          ops.withName(getInputNodes().get(0)).placeholder(Double.class),
          ops.constant(new int[]{1, (int) getWidth(), (int) getHeight(), 1}),
          ops.constant(new int[]{1, (int) getStrideX(), (int) getStrideY(), 1}),
          getPadding());
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  public long getHeight() {
    return height;
  }

  public void setHeight(long height) {
    this.height = height;
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

  public String getPadding() {
    return padding;
  }

  public void setPadding(String padding) {
    this.padding = padding;
  }

  public long getStrideX() {
    return strideX;
  }

  public void setStrideX(long strideX) {
    this.strideX = strideX;
  }

  public long getStrideY() {
    return strideY;
  }

  public void setStrideY(long strideY) {
    this.strideY = strideY;
  }

  @Nullable
  @Override
  public String getSummaryOut() {
    return null;
  }

  public long getWidth() {
    return width;
  }

  public void setWidth(long width) {
    this.width = width;
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MaxPoolLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxPoolLayer(json, rs);
  }

  @Nonnull
  private static RefMap<String, Tensor> defaultStates() {
    return new RefHashMap<>();
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    assert json != null;
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("width", width);
    json.addProperty("height", height);
    json.addProperty("padding", padding);
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MaxPoolLayer addRef() {
    return (MaxPoolLayer) super.addRef();
  }

  @Nonnull
  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    return new HashSet<>();
  }
}
