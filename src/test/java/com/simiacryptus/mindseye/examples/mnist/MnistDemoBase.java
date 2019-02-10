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

package com.simiacryptus.mindseye.examples.mnist;

import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayerBase;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.test.LabeledObject;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.GraphDef;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public abstract class MnistDemoBase {
  private static final Logger log = LoggerFactory.getLogger(MnistDemoBase.class);
  @Test
  public void test() throws Exception {
    File tensorboardLocation = new File("tensorboard/tflayer_TFLayer").getAbsoluteFile();
    File[] listFiles = tensorboardLocation.getParentFile().listFiles();
    if (null != listFiles) Arrays.stream(listFiles).forEach(file -> {
      System.out.println("Delete: " + file);
      file.delete();
    });
    byte[] graphDef = getGraphDef();
    if(null != graphDef) {
      TFLayerBase.eventWriter = new TensorboardEventWriter(tensorboardLocation, GraphDef.parseFrom(graphDef));
    }
    File reportFile = new File(String.format("target/reports/%s/%s/test",
        getClass().getSimpleName(), new SimpleDateFormat("yyyyMMddHHmm").format(new Date())));
    MarkdownNotebookOutput log = new MarkdownNotebookOutput(reportFile, true);
    try(CodeUtil.LogInterception ignored = CodeUtil.intercept(log, ReferenceCountingBase.class.getCanonicalName())) {
      run(log);
    } finally {
      log.close();
    }
    if(null != TFLayerBase.eventWriter) {
      TFLayerBase.eventWriter.close();
      TFLayerBase.eventWriter = null;
    }
    if(null != graphDef) {
      TFUtil.launchTensorboard(tensorboardLocation.getParentFile(), x -> x.waitFor(1, TimeUnit.HOURS));
    }
  }

  protected abstract byte[] getGraphDef();

  public void run(@Nonnull NotebookOutput log) {

    final Tensor[][] trainingData = MNIST.trainingDataStream().map(labeledObject1 -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject1.label);
      categoryTensor.set(category, 1);
      return new Tensor[]{labeledObject1.data, categoryTensor};
    }).toArray(i1 -> new Tensor[i1][]);

    log.h1("Model");
    final Layer recognitionNetwork = buildModel(log);

    log.h1("Training");
    @Nonnull final List<Step> history = new ArrayList<>();
    @Nonnull final TrainingMonitor monitor = new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        MnistDemoBase.log.info(msg);
//        if(null != TFLayerBase.eventWriter) {
//          try {
//            TFLayerBase.eventWriter.write(LogMessage.newBuilder().setLevel(LogMessage.Level.INFO).setMessage(msg).build());
//          } catch (IOException e) {
//            throw new RuntimeException(e);
//          }
//        }
        super.log(msg);
      }

      @Override
      public void onStepComplete(final Step currentPoint) {
        if(null != TFLayerBase.eventWriter) {
          TFLayerBase.eventWriter.setStep(currentPoint.iteration);
        }
        history.add(currentPoint);
        super.onStepComplete(currentPoint);
      }
    };
    log.eval(() -> {
      EntropyLossLayer loss = new EntropyLossLayer();
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(recognitionNetwork, loss);
      loss.freeRef();
      @Nonnull final Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
      supervisedNetwork.freeRef();
      double result = new IterativeTrainer(trainable)
          .setMonitor(monitor)
          .setOrientation(new LBFGS())
          .setLineSearchFactory(n -> new ArmijoWolfeSearch().setAlpha(1e-4))
//          .setLineSearchFactory(n -> new QuadraticSearch().setCurrentRate(1e-4))
          .setTimeout(60, TimeUnit.SECONDS)
          .setMaxIterations(200)
          .setIterationsPerSample(20)
          .runAndFree();
      trainable.freeRef();
      return result;
    });
    if (!history.isEmpty()) {
      log.eval(() -> {
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }

    @Nonnull final String modelName = "model.json";
    log.p("Saved model as " + log.file(recognitionNetwork.getJson().toString(), modelName, modelName));

    if(null != TFLayerBase.eventWriter) {
      try {
        TFLayerBase.eventWriter.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      TFLayerBase.eventWriter = null;
    }

    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(() -> {
      return MNIST.validationDataStream().mapToDouble(labeledObject ->
          predict(recognitionNetwork, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
          .average().getAsDouble() * 100;
    });

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(() -> {
      @Nonnull final TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        final int actualCategory = parse(labeledObject.label);
        @Nullable final double[] predictionSignal = recognitionNetwork.eval(labeledObject.data).getData().get(0).getData();
        final int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
        if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
        @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
        row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
        row.put("Prediction", Arrays.stream(predictionList).limit(3)
            .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
            .reduce((a, b) -> a + ", " + b).get());
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });

  }

  protected abstract Layer buildModel(@Nonnull NotebookOutput log);

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Tensor tensor = network.eval(labeledObject.data).getDataAndFree().getAndFree(0);
    int[] prediction = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -tensor.getData()[i])).mapToInt(x -> x).toArray();
    tensor.freeRef();
    return prediction;
  }
}
