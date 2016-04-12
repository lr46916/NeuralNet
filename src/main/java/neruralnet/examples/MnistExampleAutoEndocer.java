package neruralnet.examples;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import neruralnet.NeuralNetUtil;
import neruralnet.StatelessLayeredNeuralNet;
import neruralnet.examples.dataloader.DigitImage;
import neruralnet.examples.dataloader.DigitImageLoadingService;
import neruralnet.function.activation.ActivationFun;
import neruralnet.function.activation.ActivationFunBP;
import neruralnet.function.distribution.Distribution;
import neruralnet.function.distribution.impl.NormalDist;
import neruralnet.layer.BPLayer;
import neruralnet.layer.impl.FullyConnected;
import neruralnet.learning.Autoencoder;
import neruralnet.learning.BackPropagation;
import neruralnet.sample.DataSample;

public class MnistExampleAutoEndocer {
	private static List<DataSample> loadData(String labelsPath, String imgPath) throws IOException {
		List<DigitImage> images = DigitImageLoadingService.loadDigitImages(labelsPath, imgPath);
		List<DataSample> trainData = new ArrayList<>(images.size());

		for (int i = 0; i < images.size(); ++i) {
			double[] inputs = new double[784];
			for (int j = 0; j < inputs.length; ++j) {
				byte tmp = images.get(i).imageData[j];
				inputs[j] = Byte.toUnsignedInt(tmp) / 255.;
			}
			double[] outputs = new double[10];
			outputs[images.get(i).label] = 1;
			trainData.add(new DataSample(inputs, outputs));
		}
		return trainData;
	}

	public static void main(String[] args) throws IOException {
		List<DataSample> trainData = loadData("lib/train-labels.idx1-ubyte", "lib/train-images.idx3-ubyte");

		Distribution uniformSmall = (n) -> {
			Random rand = new Random();
			double[] res = new double[n];
			for (int i = 0; i < n; ++i) {
				res[i] = 2 * rand.nextDouble() - 1;
			}
			return res;
		};

		StatelessLayeredNeuralNet ffn = NeuralNetUtil.createNetwork("784x30x10", uniformSmall, ActivationFun.sigmoid,
				ActivationFun.sigmoid);

		BPLayer[] layers = new BPLayer[ffn.getLayers().length];
		ActivationFunBP[] activations = new ActivationFunBP[ffn.getActivations().length];

		for (int i = 0; i < layers.length; ++i) {
			layers[i] = (BPLayer) ffn.getLayers()[i];
			activations[i] = (ActivationFunBP) ffn.getActivations()[i];
		}

		StatelessLayeredNeuralNet ffn2 = NeuralNetUtil.createNetwork("784x30x784", uniformSmall, ActivationFun.sigmoid,
				ActivationFun.sigmoid);
		BPLayer[] layers2 = new BPLayer[ffn.getLayers().length];
		ActivationFunBP[] activations2 = new ActivationFunBP[ffn.getActivations().length];

		for (int i = 0; i < layers.length; ++i) {
			layers2[i] = (BPLayer) ffn2.getLayers()[i];
			activations2[i] = (ActivationFunBP) ffn2.getActivations()[i];
		}

		BackPropagation backprop2 = new BackPropagation(layers2, activations2);

		List<DataSample> samples2 = new ArrayList<>();

		for (DataSample s : trainData.subList(0, 1000)) {
			samples2.add(new DataSample(s.inputs, Arrays.copyOf(s.inputs, s.inputs.length)));
		}

		backprop2.trainBackpropagation(samples2, 30, 10, 1, 0.1);

	}

	private static double performance(StatelessLayeredNeuralNet ffn, List<DataSample> data) {

		int hit = 0;
		double[] outputs = new double[ffn.getNumberOfOutputs()];
		for (DataSample sample : data) {
			ffn.apply(sample.inputs, outputs);
			int maxInd = 0;
			double maxVal = outputs[0];
			for (int i = 1; i < outputs.length; ++i) {
				if (maxVal < outputs[i]) {
					maxInd = i;
					maxVal = outputs[i];
				}
			}
			if (sample.outputs[maxInd] == 1) {
				hit++;
			}
		}
		return (double) hit / data.size();
	}
}
