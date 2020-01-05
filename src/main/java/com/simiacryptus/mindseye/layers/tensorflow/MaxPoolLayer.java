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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

public @RefAware
class MaxPoolLayer extends TFLayerBase {

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

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      RefList<String> temp_29_0001 = getInputNodes();
      RefList<Long> temp_29_0002 = RefArrays.asList(1L, getWidth(), getHeight(),
          1L);
      RefList<Long> temp_29_0003 = RefArrays.asList(1L, getStrideX(),
          getStrideY(), 1L);
      ops.withName(getOutputNode()).maxPool(ops.withName(temp_29_0001.get(0)).placeholder(Double.class), temp_29_0002,
          temp_29_0003, getPadding());
      if (null != temp_29_0003)
        temp_29_0003.freeRef();
      if (null != temp_29_0002)
        temp_29_0002.freeRef();
      if (null != temp_29_0001)
        temp_29_0001.freeRef();
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

  @Override
  public RefList<String> getInputNodes() {
    return RefArrays.asList("input");
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  public String getPadding() {
    return padding;
  }

  public MaxPoolLayer setPadding(String padding) {
    this.padding = padding;
    return this.addRef();
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

  public static @SuppressWarnings("unused")
  MaxPoolLayer[] addRefs(MaxPoolLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayer::addRef)
        .toArray((x) -> new MaxPoolLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MaxPoolLayer[][] addRefs(MaxPoolLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxPoolLayer::addRefs)
        .toArray((x) -> new MaxPoolLayer[x][]);
  }

  private static RefMap<String, Tensor> defaultStates() {
    return new RefHashMap<>();
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

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MaxPoolLayer addRef() {
    return (MaxPoolLayer) super.addRef();
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    return new RefHashSet<>();
  }
}
