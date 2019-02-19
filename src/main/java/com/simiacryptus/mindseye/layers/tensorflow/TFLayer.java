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
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import java.util.*;

public class TFLayer extends TFLayerBase {

  private final byte[] graphDef;
  private boolean isFloat = false;
  private boolean singleBatch = false;
  private String outputNode;
  private List<String> inputNodes;
  private String summaryOut = "";

  public boolean isSingleBatch() {
    return singleBatch;
  }

  public TFLayer setSingleBatch(boolean singleBatch) {
    this.singleBatch = singleBatch;
    return this;
  }

  public TFLayer(byte[] graphDef, Map<String, com.simiacryptus.mindseye.lang.Tensor> states, String output, String... input) {
    super(states);
    this.setOutputNode(output);
    setInputNodes(Arrays.asList(input));
    this.graphDef = graphDef;
  }

  @Nonnull
  public static TFLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new TFLayer(json, rs);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("graphDef", Base64.getEncoder().encodeToString(graphDef));
    json.addProperty("singleBatch", isSingleBatch());
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

  @Override
  protected boolean floatInputs(String key) {
    return isFloat();
  }

  public TFLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json,rs);
    graphDef = Base64.getDecoder().decode(json.get("graphDef").getAsString());
    setSingleBatch(json.get("singleBatch").getAsBoolean());
    setFloat(json.get("isFloat").getAsBoolean());
    setOutputNode(json.get("output").getAsString());
    JsonArray jsonArray = json.get("input").getAsJsonArray();
    ArrayList<String> inputNodes = new ArrayList<>();
    for (int i = 0; i < jsonArray.size(); i++) {
      inputNodes.add(jsonArray.get(i).getAsString());
    }
    setInputNodes(inputNodes);
    setSummaryOut(json.get("summaryOut").getAsString());
  }

  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    JsonArray dataKeys = json.get("dataKeys").getAsJsonArray();
    HashSet<String> hashSet = new HashSet<>();
    for (int i = 0; i < dataKeys.size(); i++) {
      hashSet.add(dataKeys.get(i).getAsString());
    }
    return hashSet;
  }

  @Override
  public GraphDef getGraphDef() {
    try {
      return GraphDef.parseFrom(this.graphDef);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }
  public String getOutputNode() {
    return outputNode;
  }

  public List<String> getInputNodes() {
    return inputNodes;
  }

  public TFLayer setOutputNode(String outputNode) {
    this.outputNode = outputNode;
    return this;
  }

  public TFLayer setInputNodes(List<String> inputNodes) {
    this.inputNodes = inputNodes;
    return this;
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
}
