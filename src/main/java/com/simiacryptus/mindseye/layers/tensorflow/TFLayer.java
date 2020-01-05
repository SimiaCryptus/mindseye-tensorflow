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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Base64;
import java.util.Map;

public @RefAware
class TFLayer extends TFLayerBase {

  private final byte[] graphDef;
  private boolean isFloat = false;
  private String outputNode;
  private RefList<String> inputNodes;
  private String summaryOut = "";

  public TFLayer(byte[] graphDef, RefMap<String, Tensor> states, String output,
                 String... input) {
    super(states);
    this.setOutputNode(output);
    setInputNodes(RefArrays.asList(input));
    this.graphDef = graphDef;
  }

  public TFLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    graphDef = Base64.getDecoder().decode(json.get("graphDef").getAsString());
    setFloat(json.get("isFloat").getAsBoolean());
    setOutputNode(json.get("output").getAsString());
    JsonArray jsonArray = json.get("input").getAsJsonArray();
    RefArrayList<String> inputNodes = new RefArrayList<>();
    for (int i = 0; i < jsonArray.size(); i++) {
      inputNodes.add(jsonArray.get(i).getAsString());
    }
    setInputNodes(inputNodes);
    setSummaryOut(json.get("summaryOut").getAsString());
  }

  @Override
  public GraphDef getGraphDef() {
    try {
      return GraphDef.parseFrom(this.graphDef);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  public RefList<String> getInputNodes() {
    return inputNodes;
  }

  public void setInputNodes(RefList<String> inputNodes) {
    this.inputNodes = inputNodes;
  }

  public String getOutputNode() {
    return outputNode;
  }

  public void setOutputNode(String outputNode) {
    this.outputNode = outputNode;
  }

  public String getSummaryOut() {
    return summaryOut;
  }

  public TFLayer setSummaryOut(String summaryOut) {
    this.summaryOut = summaryOut;
    return this;
  }

  public boolean isFloat() {
    return isFloat;
  }

  public TFLayer setFloat(boolean aFloat) {
    isFloat = aFloat;
    return this;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static TFLayer fromJson(@Nonnull final JsonObject json,
                                 Map<CharSequence, byte[]> rs) {
    return new TFLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  TFLayer[] addRefs(TFLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayer::addRef).toArray((x) -> new TFLayer[x]);
  }

  public static @SuppressWarnings("unused")
  TFLayer[][] addRefs(TFLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayer::addRefs)
        .toArray((x) -> new TFLayer[x][]);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("graphDef", Base64.getEncoder().encodeToString(graphDef));
    JsonArray array = new JsonArray();
    getWeights().keySet().forEach(array::add);
    json.add("dataKeys", array);
    json.addProperty("output", getOutputNode());
    JsonArray jsonArray = new JsonArray();
    getInputNodes().forEach(jsonArray::add);
    json.add("input", jsonArray);
    json.addProperty("isFloat", isFloat());
    json.addProperty("summaryOut", getSummaryOut());
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  TFLayer addRef() {
    return (TFLayer) super.addRef();
  }

  @Override
  protected boolean floatInputs(String key) {
    return isFloat();
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    JsonArray dataKeys = json.get("dataKeys").getAsJsonArray();
    RefHashSet<String> hashSet = new RefHashSet<>();
    for (int i = 0; i < dataKeys.size(); i++) {
      hashSet.add(dataKeys.get(i).getAsString());
    }
    return hashSet;
  }
}
