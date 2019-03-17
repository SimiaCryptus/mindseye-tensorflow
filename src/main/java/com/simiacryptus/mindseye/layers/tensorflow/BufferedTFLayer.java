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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.tensorflow.Tensor;

import javax.annotation.Nullable;
import java.util.*;

public class BufferedTFLayer extends LayerBase {
  private TFLayerBase tfLayerBase;
  private final TFLayerBase parent;
  private final TFLayerBase.TFSession tfsession;

  BufferedTFLayer(TFLayerBase tfLayerBase) {
    this.tfLayerBase = tfLayerBase;
    this.parent = tfLayerBase;
    this.tfsession = tfLayerBase.new TFSession(true) {
      public void incrementWeights(DeltaSet<UUID> deltaBuffer, String weightNodeName, org.tensorflow.Tensor<Number> doubleTensor) {
        deltas.computeIfAbsent(weightNodeName, k->new ArrayList<>()).add(doubleTensor);
      }
    };
  }

  private final HashMap<String, ArrayList<Tensor<Number>>> deltas = new HashMap<>();

  public void writeDeltas(DeltaSet<UUID> deltaBuffer) {
    assertAlive();
    deltas.forEach((st,v)->{
      UUID key = UUID.nameUUIDFromBytes((tfLayerBase.getId() + "_" + st).getBytes());
      Delta<UUID> uuidDelta = deltaBuffer.get(key, tfLayerBase.getWeights().get(st));
      ArrayList<Tensor<Number>> copy = new ArrayList<>(v);
      v.clear();
      try(Tensor<Number> tensor = TensorflowUtil.add(copy.stream())) {
        com.simiacryptus.mindseye.lang.Tensor t = TFIO.getTensor(tensor.expect(Double.class));
        uuidDelta.addInPlace(t.getData());
        t.freeRef();
        uuidDelta.freeRef();
      }
    });
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return parent.getJson(resources, dataSerializer);
  }

  @Nullable
  @Override
  public List<double[]> state() {
    return parent.state();
  }

  @Nullable
  @Override
  public Result evalAndFree(Result... inputs) {
    return parent.evalAndFree(tfsession, inputs);
  }

  @Override
  protected void _free() {
    tfsession.freeRef();
    super._free();
  }
}
