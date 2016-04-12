package neruralnet.learning;

import java.util.ArrayList;
import java.util.List;

import neruralnet.function.activation.ActivationFunBP;
import neruralnet.layer.BPLayer;
import neruralnet.sample.DataSample;

public class Autoencoder {

	private BPLayer layer;
	private ActivationFunBP activation;
	private BPLayer mockOutputLayer;
	private ActivationFunBP mockLayerActivation;
	private List<double[]> inputs;

	public Autoencoder(BPLayer layer, ActivationFunBP activation, BPLayer mockOutputLayer,
			ActivationFunBP mockLayerActivation, List<double[]> inputs) {
		super();
		this.layer = layer;
		this.activation = activation;
		this.mockOutputLayer = mockOutputLayer;
		this.mockLayerActivation = mockLayerActivation;
		this.inputs = inputs;
		if (mockOutputLayer.getNumberOfOutputs() != inputs.get(0).length) {
			throw new IllegalArgumentException(
					"In order to run autoencoding mock output layer MUST have same number of outputs as dimensions of input features.");
		}
	}

	public List<double[]> encode(int epochs, int miniBatchSize, int epochOutput, double learningRate) {
		List<DataSample> samples = new ArrayList<>(this.inputs.size());
		for (double[] input : inputs) {
			samples.add(new DataSample(input, input));
		}
		BackPropagation backPropagation = new BackPropagation(new BPLayer[] { layer, mockOutputLayer },
				new ActivationFunBP[] { activation, mockLayerActivation });
		backPropagation.trainBackpropagation(samples, epochs, miniBatchSize, epochOutput, learningRate);
		
		List<double[]> results = new ArrayList<>();
		for(double[] input : inputs) {
			double[] outputs = new double[layer.getNumberOfOutputs()];
			layer.apply(input, outputs);
			results.add(outputs);
		}
		return results;
	}
}
