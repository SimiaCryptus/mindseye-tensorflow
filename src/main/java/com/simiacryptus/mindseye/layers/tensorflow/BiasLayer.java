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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.util.Util;
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

public class BiasLayer extends TFLayerBase {

  public BiasLayer(int... intputDims) {
    super(defaultStates(intputDims));
  }

  public BiasLayer(@Nonnull JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @Override
  public GraphDef getGraphDef() {
    try (Graph graph = new Graph()) {
      Ops ops = Ops.create(graph);
      ops.withName(getOutputNode()).math.add(ops.withName("bias").placeholder(Double.class),
          ops.withName(getInputNodes().get(0)).placeholder(Double.class));
      return GraphDef.parseFrom(graph.toGraphDef());
    } catch (InvalidProtocolBufferException e) {
      throw Util.throwException(e);
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
  public static BiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasLayer(json, rs);
  }


  @Nonnull
  private static RefMap<String, Tensor> defaultStates(int[] intputDims) {
    RefHashMap<String, Tensor> map = new RefHashMap<>();
    Tensor temp_15_0001 = new Tensor(intputDims);
    temp_15_0001.setByCoord(c -> {
      return 0;
    });
    RefUtil.freeRef(map.put("bias", temp_15_0001.addRef()));
    temp_15_0001.freeRef();
    return map;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BiasLayer addRef() {
    return (BiasLayer) super.addRef();
  }

  @Nonnull
  @Override
  protected Set<String> getDataKeys(JsonObject json) {
    Set<String> hashSet = new HashSet<>();
    hashSet.add("bias");
    return hashSet;
  }

}
