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
import com.simiacryptus.ref.wrappers.RefHashMap;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.LRN;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

public class LRNLayer extends TFLayerBase {

  private long radius = 5L;
  private float beta = .5f;
  private float alpha = 1.0f;
  private float bias = 1.0f;

  public LRNLayer() {
    super(new RefHashMap<>());
  }

  public LRNLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    setRadius((json.get("width").getAsInt() - 1) / 2);
    setAlpha((float) (json.get("alpha").getAsDouble() / (double) (getRadius() * 2 + 1)));
    setBeta((float) json.get("beta").getAsDouble());
    setBias((float) json.get("k").getAsDouble());
  }

  public float getAlpha() {
    return alpha;
  }

  public void setAlpha(float alpha) {
    this.alpha = alpha;
  }

  public float getBeta() {
    return beta;
  }

  public void setBeta(float beta) {
    this.beta = beta;
  }

  public float getBias() {
    return bias;
  }

  public void setBias(float bias) {
    this.bias = bias;
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).lRN(ops.withName(getInputNodes().get(0)).placeholder(Float.class),
          LRN.depthRadius(getRadius()).beta(getBeta()).alpha(getAlpha()).bias(getBias()));
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
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

  public long getRadius() {
    return radius;
  }

  public void setRadius(long radius) {
    this.radius = radius;
  }

  @Nullable
  @Override
  public String getSummaryOut() {
    return null;
  }

  public boolean isSingleBatch() {
    return false;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static LRNLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LRNLayer(json, rs);
  }


  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    long width = getRadius() * 2 + 1;
    assert json != null;
    json.addProperty("width", width);
    json.addProperty("alpha", getAlpha() * (double) width);
    json.addProperty("beta", getBeta());
    json.addProperty("k", getBias());
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LRNLayer addRef() {
    return (LRNLayer) super.addRef();
  }

  @Override
  protected boolean floatInputs(String key) {
    return true;
  }

  @Nonnull
  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    return new HashSet<>();
  }
}
