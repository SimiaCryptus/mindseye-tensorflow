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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.Summary;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import javax.annotation.Nullable;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class TFLayerBase extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(TFLayer.class);
  public static TensorboardEventWriter eventWriter = null;

  private final Map<String, Tensor> weights = new HashMap<>();

  public TFLayerBase(JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    Set<String> dataKeys = getDataKeys(json);
    for (String key : dataKeys) {
      this.getWeights().put(key, Tensor.fromJson(json.get(key), rs));
    }
  }

  public TFLayerBase(Map<String, Tensor> states) {
    this.getWeights().putAll(states);
  }

  public abstract GraphDef getGraphDef();

  public abstract List<String> getInputNodes();

  public abstract String getOutputNode();

  public abstract String getSummaryOut();

  public Map<String, Tensor> getWeights() {
    return weights;
  }

  @NotNull
  public TFLayer asConstLayer() {
    return new TFLayer(constGraph().toByteArray(), new HashMap<>(), getOutputNode(),
        getInputNodes().toArray(new String[]{}));
  }

  public @NotNull GraphDef constGraph() {
    return TFUtil.implantConstants(getGraphDef(), getWeights());
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = getJsonStub();
    for (Map.Entry<String, Tensor> entry : getWeights().entrySet()) {
      json.add(entry.getKey(), entry.getValue().getJson(resources, dataSerializer));
    }
    return json;
  }

  @Nullable
  @Override
  public List<double[]> state() {
    return getWeights().values().stream().map(x -> x.getData()).collect(Collectors.toList());
  }

  @Nullable
  @Override
  public Result eval(Result... inputs) {
    TFSession tfsession = new TFSession();
    return eval(tfsession, inputs);
  }

  public void close() {
  }

  public boolean invertWeights() {
    return true;
  }

  public @NotNull GraphDef getConstGraph(GraphDef graphDef) {
    return TFUtil.implantConstants(graphDef, getWeights());
  }

  @NotNull
  Result eval(TFSession tfsession, Result... inputs) {
    List<String> stateNames = getWeights().keySet().stream().collect(Collectors.toList());
    Session.Runner runner = tfsession.session.runner();
    ArrayList<org.tensorflow.Tensor<?>> tensors = new ArrayList<>();
    getWeights().forEach((nodeName, data) -> {
      org.tensorflow.@NotNull Tensor<? extends Number> tensor;
      if (floatInputs(nodeName)) {
        tensor = TFIO.getFloatTensor(data, invertWeights());
      } else {
        tensor = TFIO.getDoubleTensor(data, invertWeights());
      }
      runner.feed(nodeName, tensor);
      tensors.add(tensor);
    });
    for (int i = 0; i < getInputNodes().size(); i++) {
      String inputNode = getInputNodes().get(i);
      TensorList data = inputs[i].getData();
      org.tensorflow.@NotNull Tensor<? extends Number> tensor;
      if (floatInputs(inputNode)) {
        tensor = TFIO.getFloatTensor(data);
      } else {
        tensor = TFIO.getDoubleTensor(data);
      }
      runner.feed(inputNode, tensor);
      tensors.add(tensor);
    }
    runner.fetch(getOutputNode());
    boolean summaryOut = null != eventWriter && null != getSummaryOut() && !getSummaryOut().isEmpty();
    int fwdFetches;
    if (summaryOut) {
      fwdFetches = 2;
      runner.fetch(getSummaryOut());
    } else {
      fwdFetches = 1;
    }
    Session.Run fwd;
    try {
      fwd = runner.runAndFetchMetadata();
    } catch (IllegalArgumentException e) {
      throw e;
    }
    TensorArray resultData;
    {
      org.tensorflow.Tensor<?> tensor = fwd.outputs.get(0);
      resultData = TFIO.getTensorList(tensor);
      tensors.add(tensor);
    }
    if (summaryOut) {
      final Summary summary;
      try {
        summary = Summary.parseFrom(fwd.outputs.get(1).expect(String.class).bytesValue());
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      try {
        if (null != eventWriter)
          eventWriter.write(summary);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return new Result(resultData, ((deltaBuffer, deltaSignal) -> {
      ArrayList<org.tensorflow.Tensor<?>> feedbacktensors = new ArrayList<>();
      Output<?>[] gradients = tfsession.getGradients();
      String deltaOperation = getOutputNode() + "_delta";
      if (floatInputs(deltaOperation)) {
        org.tensorflow.Tensor<Float> tensor = TFIO.getFloatTensor(deltaSignal);
        runner.feed(deltaOperation, tensor);
        feedbacktensors.add(tensor);
      } else {
        org.tensorflow.Tensor<Double> tensor = TFIO.getDoubleTensor(deltaSignal);
        runner.feed(deltaOperation, tensor);
        feedbacktensors.add(tensor);
      }
      Arrays.stream(gradients).forEach(runner::fetch);
      Session.Run back = runner.runAndFetchMetadata();
      for (int i = 0; i < inputs.length; i++) {
        org.tensorflow.Tensor<?> tensor = back.outputs.get(fwdFetches + i);
        inputs[i].getAccumulator().accept(deltaBuffer, TFIO.getTensorList(tensor));
        feedbacktensors.add(tensor);
      }
      for (int i = 0; i < stateNames.size(); i++) {
        String weightNodeName = stateNames.get(i);
        Delta<UUID> uuidDelta = deltaBuffer.get(UUID.nameUUIDFromBytes((getId() + "_" + weightNodeName).getBytes()),
            getWeights().get(weightNodeName));
        org.tensorflow.Tensor<Number> numberTensor = (org.tensorflow.Tensor<Number>) back.outputs
            .get(i + fwdFetches + getInputNodes().size());
        Tensor t;
        if (numberTensor.dataType() == DataType.FLOAT) {
          t = TFIO.getTensor(numberTensor.expect(Float.class), invertWeights());
        } else {
          t = TFIO.getTensor(numberTensor.expect(Double.class), invertWeights());
        }
        uuidDelta.addInPlace(t.getData());
      }
      feedbacktensors.stream().forEach(org.tensorflow.Tensor::close);
    })) {
      @Override
      protected void _free() {
        tensors.stream().forEach(org.tensorflow.Tensor::close);
        super._free();
      }
    };
  }

  protected abstract Set<String> getDataKeys(JsonObject json);

  protected boolean floatInputs(String key) {
    return false;
  }

  @Override
  protected void _free() {
    close();
    super._free();
  }

  class TFSession extends ReferenceCountingBase {
    public final Graph graph;
    public final Singleton<Output<?>[]> outputSingleton = new Singleton<>();
    public final Session session;

    public TFSession() {
      this.graph = new Graph();
      GraphDef graphDef = getGraphDef();
      TensorflowUtil.validate(graphDef);
      graph.importGraphDef(graphDef.toByteArray());
      this.session = new Session(graph);
    }

    public Output<?>[] getGradients() {
      return outputSingleton.getOrInit(() -> {
        List<String> stateNames = getWeights().keySet().stream().collect(Collectors.toList());
        Ops ops = Ops.create(graph);
        String deltaOpName = getOutputNode() + "_delta";
        Class<? extends Number> dtype = floatInputs(deltaOpName) ? Float.class : Double.class;
        ops.withName(deltaOpName).placeholder(dtype, Placeholder.shape(Shape.unknown()));
        return graph.addGradients("gradient", new Output[]{TensorflowUtil.find(graph, getOutputNode()).output(0)},
            Stream.concat(getInputNodes().stream(), stateNames.stream())
                .map(n -> TensorflowUtil.find(graph, n).output(0)).toArray(i -> new Output[i]),
            new Output[]{TensorflowUtil.find(graph, deltaOpName).output(0)});
      });
    }

    @Override
    protected void _free() {
      new Thread(() -> {
        session.close();
        graph.close();
      }).start();
      super._free();
    }
  }

}
