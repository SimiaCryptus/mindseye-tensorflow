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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

public class TFLayer extends TFLayerBase {

  private final byte[] graphDef;
  private boolean isFloat = false;
  private String outputNode;
  @Nullable
  private RefList<String> inputNodes;
  private String summaryOut = "";

  public TFLayer(byte[] graphDef, RefMap<String, Tensor> states, String output, String... input) {
    super(states);
    this.setOutputNode(output);
    setInputNodes(RefArrays.asList(input));
    this.graphDef = graphDef;
  }

  public TFLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    graphDef = Base64.getDecoder().decode(json.get("graphDef").getAsString());
    setFloat(json.get("isFloat").getAsBoolean());
    setOutputNode(json.get("output").getAsString());
    JsonArray jsonArray = json.get("input").getAsJsonArray();
    RefArrayList<String> inputNodes = new RefArrayList<>();
    for (int i = 0; i < jsonArray.size(); i++) {
      inputNodes.add(jsonArray.get(i).getAsString());
    }
    setInputNodes(inputNodes.addRef());
    inputNodes.freeRef();
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

  @Nullable
  public List<String> getInputNodes() {
    return inputNodes == null ? null : inputNodes.addRef();
  }

  public void setInputNodes(@Nullable RefList<String> inputNodes) {
    RefList<String> temp_02_0001 = inputNodes == null ? null : inputNodes.addRef();
    if (null != this.inputNodes)
      this.inputNodes.freeRef();
    this.inputNodes = temp_02_0001 == null ? null : temp_02_0001.addRef();
    if (null != temp_02_0001)
      temp_02_0001.freeRef();
    if (null != inputNodes)
      inputNodes.freeRef();
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

  public void setSummaryOut(String summaryOut) {
    this.summaryOut = summaryOut;
  }

  public boolean isFloat() {
    return isFloat;
  }

  public void setFloat(boolean aFloat) {
    isFloat = aFloat;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static TFLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new TFLayer(json, rs);
  }


  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    assert json != null;
    json.addProperty("graphDef", Base64.getEncoder().encodeToString(graphDef));
    JsonArray array = new JsonArray();
    RefMap<String, Tensor> temp_02_0002 = getWeights();
    assert temp_02_0002 != null;
    RefSet<String> temp_02_0003 = temp_02_0002.keySet();
    temp_02_0003.forEach(array::add);
    temp_02_0003.freeRef();
    temp_02_0002.freeRef();
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
    super._free();
    if (null != inputNodes)
      inputNodes.freeRef();
    inputNodes = null;
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TFLayer addRef() {
    return (TFLayer) super.addRef();
  }

  @Override
  protected boolean floatInputs(String key) {
    return isFloat();
  }

  @Nonnull
  @Override
  protected Set<String> getDataKeys(@Nonnull JsonObject json) {
    JsonArray dataKeys = json.get("dataKeys").getAsJsonArray();
    Set<String> hashSet = new HashSet<>();
    for (int i = 0; i < dataKeys.size(); i++) {
      hashSet.add(dataKeys.get(i).getAsString());
    }
    return hashSet;
  }
}
