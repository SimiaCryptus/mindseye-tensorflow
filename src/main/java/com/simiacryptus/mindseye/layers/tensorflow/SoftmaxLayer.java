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
import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.op.Ops;

import javax.annotation.Nonnull;
import java.util.*;

public class SoftmaxLayer extends TFLayerBase {

  public boolean isSingleBatch() {
    return false;
  }

  public SoftmaxLayer() {
    super(new HashMap<>());
  }

  @Nonnull
  public static SoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxLayer(json, rs);
  }

  public SoftmaxLayer(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
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
      ops.withName(getOutputNode()).softmax(
          ops.withName(getInputNodes().get(0)).placeholder(Double.class)
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

}
