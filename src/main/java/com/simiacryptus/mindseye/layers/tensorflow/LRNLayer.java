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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.LRN;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

public @RefAware
class LRNLayer extends TFLayerBase {

  private long radius = 5L;
  private float beta = .5f;
  private float alpha = 1.0f;
  private float bias = 1.0f;

  public LRNLayer() {
    super(new RefHashMap<>());
  }

  public LRNLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    RefUtil.freeRef(setRadius((json.get("width").getAsInt() - 1) / 2));
    RefUtil
        .freeRef(setAlpha((float) (json.get("alpha").getAsDouble() / ((double) (getRadius() * 2 + 1)))));
    setBeta((float) json.get("beta").getAsDouble());
    RefUtil.freeRef(setBias((float) json.get("k").getAsDouble()));
  }

  public float getAlpha() {
    return alpha;
  }

  public LRNLayer setAlpha(float alpha) {
    this.alpha = alpha;
    return this.addRef();
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

  public LRNLayer setBias(float bias) {
    this.bias = bias;
    return this.addRef();
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      RefList<String> temp_25_0001 = getInputNodes();
      ops.withName(getOutputNode()).lRN(ops.withName(temp_25_0001.get(0)).placeholder(Float.class),
          LRN.depthRadius(getRadius()).beta(getBeta()).alpha(getAlpha()).bias(getBias()));
      if (null != temp_25_0001)
        temp_25_0001.freeRef();
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public RefList<String> getInputNodes() {
    return RefArrays.asList("input");
  }

  @Override
  public String getOutputNode() {
    return "output";
  }

  public long getRadius() {
    return radius;
  }

  public LRNLayer setRadius(long radius) {
    this.radius = radius;
    return this.addRef();
  }

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

  public static @SuppressWarnings("unused")
  LRNLayer[] addRefs(LRNLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayer::addRef)
        .toArray((x) -> new LRNLayer[x]);
  }

  public static @SuppressWarnings("unused")
  LRNLayer[][] addRefs(LRNLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LRNLayer::addRefs)
        .toArray((x) -> new LRNLayer[x][]);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    long width = getRadius() * 2 + 1;
    json.addProperty("width", width);
    json.addProperty("alpha", getAlpha() * ((double) width));
    json.addProperty("beta", getBeta());
    json.addProperty("k", getBias());
    return json;
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LRNLayer addRef() {
    return (LRNLayer) super.addRef();
  }

  @Override
  protected boolean floatInputs(String key) {
    return true;
  }

  @Override
  protected RefSet<String> getDataKeys(JsonObject json) {
    return new RefHashSet<>();
  }
}
