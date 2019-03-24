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

public class Conv2DLayer extends TFLayerBase {

  private final Class<Double> dtype = Double.class;
  private String padding = "SAME";
  private int strideX = 1;
  private int strideY = 1;

  public Conv2DLayer(int... intputDims) {
    super(defaultStates(intputDims));
  }

  public Conv2DLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    padding = json.get("padding").getAsString();
  }

  private static Map<String, Tensor> defaultStates(int[] intputDims) {
    HashMap<String, Tensor> map = new HashMap<>();
    map.put("kernel", new Tensor(intputDims).setByCoord(c -> {
      return 0;
    }));
    return map;
  }

  @Nonnull
  public static Conv2DLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new Conv2DLayer(json, rs);
  }

  @Override
  protected boolean floatInputs(String key) {
    return false;
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("padding", padding);
    return json;
  }

  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    HashSet<String> hashSet = new HashSet<>();
    hashSet.add("kernel");
    return hashSet;
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).conv2D(
          ops.withName(getInputNodes().get(0)).placeholder(dtype),
          ops.withName("kernel").placeholder(dtype),
          Arrays.asList(1L, (long) getStrideX(), (long) getStrideY(), 1L),
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

  public int getStrideX() {
    return strideX;
  }

  public Conv2DLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this;
  }

  public int getStrideY() {
    return strideY;
  }

  public Conv2DLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }

  public String getPadding() {
    return padding;
  }

  public Conv2DLayer setPadding(String padding) {
    this.padding = padding;
    return this;
  }
}
