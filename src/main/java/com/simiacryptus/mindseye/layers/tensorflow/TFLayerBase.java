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
import com.simiacryptus.lang.ref.*;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.mindseye.network.CountingResult;
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
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
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

  protected abstract Set<String> getDataKeys(JsonObject json);

  public TFLayerBase(Map<String, Tensor> states) {
    this.getWeights().putAll(states);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = getJsonStub();
    for (Map.Entry<String, Tensor> entry : getWeights().entrySet()) {
      json.add(entry.getKey(), entry.getValue().toJson(resources, dataSerializer));
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
  public Result evalAndFree(Result... inputs) {
    if (isSingleBatch() && Arrays.stream(inputs).anyMatch(x -> x.getData().length() > 1)) return new SubBatchLayer(this).evalAndFree(inputs);
    TFSession tfsession = new TFSession(this, false);
    Result result = evalAndFree(tfsession, inputs);
    tfsession.freeRef();
    return result;
  }

  @NotNull
  public BufferedTFLayer constShadow() {
    return new BufferedTFLayer(this);
  }

  @NotNull Result evalAndFree(TFSession tfsession, Result... inputs) {
    tfsession.addRef();
    List<String> stateNames = getWeights().keySet().stream().collect(Collectors.toList());
    Session.Runner runner = tfsession.session.runner();
    ArrayList<org.tensorflow.Tensor<?>> tensors = new ArrayList<>();
    getWeights().values().forEach(ReferenceCountingBase::addRef);
    if (!tfsession.constantWeights) getWeights().forEach((k, v) -> {
      long[] shape = Arrays.stream(v.getDimensions()).mapToLong(x -> x).toArray();
      double[] data = v.getData();
      org.tensorflow.Tensor<? extends Number> doubleTensor = prepareInput(k, shape, data);
      runner.feed(k, doubleTensor);
      tensors.add(doubleTensor);
    });
    List<int[]> inputDimensions = Arrays.stream(inputs).map(x -> x.getData().getDimensions()).collect(Collectors.toList());
    for (int i = 0; i < getInputNodes().size(); i++) {
      String inputNode = getInputNodes().get(i);
      TensorList data = inputs[i].getData();
      double[] buffer = TFUtil.getDoubles(data);
      long[] shape = LongStream.concat(
          LongStream.of(data.length()),
          Arrays.stream(data.getDimensions()).mapToLong(x -> x)
      ).toArray();
      org.tensorflow.Tensor<? extends Number> doubleTensor = prepareInput(inputNode, shape, buffer);
      data.freeRef();
      runner.feed(inputNode, doubleTensor);
      tensors.add(doubleTensor);
      RecycleBin.DOUBLES.recycle(buffer, buffer.length);
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
    Session.Run fwd = runner.runAndFetchMetadata();
    TensorArray resultData;
    long[] outputShape;
    {
      org.tensorflow.Tensor<?> tensor = fwd.outputs.get(0);
      outputShape = tensor.shape();
      resultData = getTensorArray(outputShape, tensor);
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
        if (null != eventWriter) eventWriter.write(summary);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return new Result(resultData, ((deltaBuffer, deltaSignal) -> {
      ArrayList<org.tensorflow.Tensor<?>> feedbacktensors = new ArrayList<>();
      Output<?>[] gradients = tfsession.getGradients();
      double[] buffer = TFUtil.getDoubles(deltaSignal);
      deltaSignal.freeRef();
      String deltaOperation = getOutputNode() + "_delta";
      if(floatInputs(deltaOperation)) {
        float[] floats = TFUtil.getFloats(buffer);
        org.tensorflow.Tensor<Float> tensor = org.tensorflow.Tensor.create(outputShape, FloatBuffer.wrap(floats));
        runner.feed(deltaOperation, tensor);
        feedbacktensors.add(tensor);
      } else {
        org.tensorflow.Tensor<Double> tensor = org.tensorflow.Tensor.create(outputShape, DoubleBuffer.wrap(buffer));
        runner.feed(deltaOperation, tensor);
        feedbacktensors.add(tensor);
      }
      RecycleBin.DOUBLES.recycle(buffer, buffer.length);
      Arrays.stream(gradients).forEach(runner::fetch);
      Session.Run back = runner.runAndFetchMetadata();
      for (int i = 0; i < inputs.length; i++) {
        org.tensorflow.Tensor<?> tensor = back.outputs.get(fwdFetches + i);
        TensorArray tensorArray = getTensorArray(tensor.shape(), tensor);
        BiConsumer<DeltaSet<UUID>, TensorList> r = inputs[i].getAccumulator();
        int prevRefs = tensorArray.currentRefCount();
        r.accept(deltaBuffer, tensorArray);
        int refDeltas = prevRefs - tensorArray.currentRefCount();
        if (refDeltas != 1 && !r.getClass().equals(CountingResult.class)) {
          throw new IllegalStateException(String.format("%s backprop finished with %s refs", r.getClass().toString(), refDeltas));
        }
        feedbacktensors.add(tensor);
      }
      for (int i = 0; i < stateNames.size(); i++) {
        String s = stateNames.get(i);
        org.tensorflow.Tensor<Number> tensor = (org.tensorflow.Tensor<Number>) back.outputs.get(i + fwdFetches + getInputNodes().size());
        tfsession.incrementWeights(deltaBuffer, s, tensor);
      }
      feedbacktensors.stream().forEach(org.tensorflow.Tensor::close);
    })) {
      @Override
      protected void _free() {
        getWeights().values().forEach(ReferenceCountingBase::freeRef);
        tensors.stream().forEach(org.tensorflow.Tensor::close);
        Arrays.stream(inputs).forEach(ReferenceCountingBase::freeRef);
        tfsession.freeRef();
        super._free();
      }
    };
  }

  @NotNull
  private TensorArray getTensorArray(long[] outputShape, org.tensorflow.Tensor<?> tensor) {
    if(tensor.dataType() == DataType.DOUBLE) {
      org.tensorflow.Tensor<Double> doubleTensor = tensor.expect(Double.class);
      double[] doubles = TFUtil.doublesToDoubles(doubleTensor);
      TensorArray resultData = TFUtil.toTensorArray(outputShape, doubles);
      RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      return resultData;
    } else if(tensor.dataType() == DataType.FLOAT) {
      org.tensorflow.Tensor<Float> doubleTensor = tensor.expect(Float.class);
      float[] doubles = TFUtil.floatsToDoubles(doubleTensor);
      TensorArray resultData = TFUtil.toTensorArray(outputShape, doubles);
      RecycleBin.FLOATS.recycle(doubles, doubles.length);
      return resultData;
    } else {
      throw new IllegalArgumentException(tensor.dataType().toString());
    }
  }

  @NotNull
  public final org.tensorflow.Tensor<? extends Number> prepareInput(String key, long[] shape, double[] data) {
    if(floatInputs(key)) {
      return org.tensorflow.Tensor.create(shape, FloatBuffer.wrap(TFUtil.getFloats(data)));
    } else {
      return org.tensorflow.Tensor.create(shape, DoubleBuffer.wrap(data));
    }
  }

  protected boolean floatInputs(String key) {
    return false;
  }


  @Override
  protected void _free() {
    close();
    getWeights().values().forEach(ReferenceCountingBase::freeRef);
    super._free();
  }

  public void close() {
  }

  public abstract GraphDef getGraphDef();

  public abstract String getSummaryOut();

  public abstract String getOutputNode();

  public abstract List<String> getInputNodes();

  public Map<String, Tensor> getWeights() {
    return weights;
  }

  protected abstract boolean isSingleBatch();

  class TFSession extends ReferenceCountingBase {
    private TFLayerBase tfLayerBase;
    public final Graph graph;
    public final Singleton<Output<?>[]> outputSingleton = new Singleton<>();
    public final Session session;
    public final boolean constantWeights;

    public TFSession(TFLayerBase tfLayerBase, boolean constantWeights) {
      this.tfLayerBase = tfLayerBase;
      this.graph = new Graph();
      this.constantWeights = constantWeights;
      GraphDef graphDef1 = tfLayerBase.getGraphDef();
      TensorflowUtil.validate(graphDef1);
      if (constantWeights) {
        graphDef1 = TFUtil.implantConstants(graphDef1, tfLayerBase.getWeights());
        TensorflowUtil.validate(graphDef1);
      }
      graph.importGraphDef(graphDef1.toByteArray());
      this.session = new Session(graph);
    }

    public void incrementWeights(DeltaSet<UUID> deltaBuffer, String s, org.tensorflow.Tensor<Number> tensor) {
      Delta<UUID> uuidDelta = deltaBuffer.get(UUID.nameUUIDFromBytes((tfLayerBase.getId() + "_" + s).getBytes()), tfLayerBase.getWeights().get(s));
      if(tensor.dataType() == DataType.DOUBLE) {
        double[] doubles = TFUtil.doublesToDoubles(tensor.expect(Double.class));
        uuidDelta.addInPlace(doubles);
        uuidDelta.freeRef();
        RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      } else {
        double[] doubles = TFUtil.toDoubles(TFUtil.floatsToDoubles(tensor.expect(Float.class)));
        uuidDelta.addInPlace(doubles);
        uuidDelta.freeRef();
        RecycleBin.DOUBLES.recycle(doubles, doubles.length);
      }
    }

    public Output<?>[] getGradients() {
      return outputSingleton.getOrInit(() -> {
        List<String> stateNames = tfLayerBase.getWeights().keySet().stream().collect(Collectors.toList());
        Ops ops = Ops.create(graph);
        String deltaOpName = tfLayerBase.getOutputNode() + "_delta";
        Class<? extends Number> dtype = floatInputs(deltaOpName)?Float.class:Double.class;
        ops.withName(deltaOpName).placeholder(
            dtype,
            Placeholder.shape(Shape.unknown())
        );
        return graph.addGradients("gradient", new Output[]{
                TensorflowUtil.find(graph, tfLayerBase.getOutputNode()).output(0)
            },
            Stream.concat(
                tfLayerBase.getInputNodes().stream(),
                stateNames.stream()
            ).map(n -> TensorflowUtil.find(graph, n).output(0)).toArray(i -> new Output[i]),
            new Output[]{
                TensorflowUtil.find(graph, deltaOpName).output(0)
            });
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
