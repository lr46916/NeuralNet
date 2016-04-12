package neruralnet.learning;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import neruralnet.NeuralNetUtil;
import neruralnet.function.activation.ActivationFun;
import neruralnet.function.activation.ActivationFunBP;
import neruralnet.layer.BPLayer;
import neruralnet.sample.DataSample;

public class BackPropagation {

	private BPLayer[] layers;
	private ActivationFunBP[] activation;
	private double[][] innerLayerOut;
	private double[][] innerLayerRawOutputs;
	private double[][] localGradient;
	private double[][][] updates;

	public BackPropagation(BPLayer[] layers, ActivationFunBP[] activation) {
		super();
		this.layers = layers;
		this.activation = activation;
		int n = layers[0].getNumberOfInputs();
		int m = layers[layers.length - 1].getNumberOfOutputs();
		this.innerLayerOut = NeuralNetUtil.allocateMetadata(layers, n, m);
		this.innerLayerRawOutputs = new double[innerLayerOut.length][];
		this.localGradient = new double[innerLayerOut.length][];
		for (int i = 0; i < innerLayerRawOutputs.length; ++i) {
			this.innerLayerRawOutputs[i] = new double[innerLayerOut[i].length];
			this.localGradient[i] = new double[innerLayerOut[i].length];
		}
		updates = new double[layers.length][][];
		for (int i = 0; i < updates.length; ++i) {
			updates[i] = new double[layers[i].getNumberOfInputs() + 1][layers[i].getNumberOfOutputs()];
		}
	}

	public void trainBackpropagation(List<DataSample> samples, int epochs, int miniBatchSize, int epochOutput,
			double learningRate) {
		int lastLayerIndex = layers.length - 1;
		for (int i = 0; i < epochs; ++i) {
//			resetUpdates(updates);
			
			//TODO find a way to shuffle values that are loaded lazily using Iterable instead of List
			Collections.shuffle(samples);
			
			Iterator<DataSample> it = samples.iterator();
			while (it.hasNext()) {
				int mini = 0;
				for(; mini < miniBatchSize && it.hasNext(); ++mini) {
					DataSample sample = it.next();
					forwardPass(layers, activation, innerLayerOut, innerLayerRawOutputs, sample.inputs);
					// output layer
					for (int j = 0; j < sample.outputs.length; ++j) {
						// calculate local gradient
						this.localGradient[lastLayerIndex][j] = activation[lastLayerIndex]
								.gradient(this.innerLayerRawOutputs[lastLayerIndex][j])
								* error(this.innerLayerOut[lastLayerIndex][j], sample.outputs[j]);
					}
					// update
					accumulateUpdate(lastLayerIndex, this.innerLayerOut[lastLayerIndex - 1],
							this.localGradient[lastLayerIndex], updates);

					for (int l = lastLayerIndex - 1; l >= 0; --l) {
						// calculate local gradient
						for (int j = 0; j < this.localGradient[l].length; ++j) {
							double sum = 0;
							for (int k : layers[l + 1].connectedNeuronsIndexes(j)) {
								sum += this.localGradient[l + 1][k] * layers[l + 1].getWeight(j, k);
							}
							this.localGradient[l][j] = activation[l].gradient(this.innerLayerRawOutputs[l][j]) * sum;
						}
						// update
						accumulateUpdate(l, l > 0 ? this.innerLayerOut[l - 1] : sample.inputs, this.localGradient[l],
								updates);
					}
				}
				applyUpdates(mini, learningRate);
			}

			if ((i + 1) % epochOutput == 0) {
				double totalError = 0;
				int total = 0;
				for (DataSample sample : samples) {
					forwardPass(layers, activation, innerLayerOut, innerLayerRawOutputs, sample.inputs);
					for (int j = 0; j < sample.outputs.length; ++j) {
						double err = error(innerLayerOut[lastLayerIndex][j], sample.outputs[j]);
						totalError += err * err;
					}
					total++;
				}
				totalError /= total;
				System.out.println("Epoch " + (i + 1) + ": train set error: " + totalError);
			}

		}

	}

	private void applyUpdates(int miniBatchSize, double learningRate) {
		for (int i = 0; i < updates.length; ++i) {
			for (int j = 0; j < updates[i].length; ++j) {
				for (int k = 0; k < updates[i][j].length; ++k) {
					layers[i].updateWeight(j, k, learningRate * updates[i][j][k] / miniBatchSize);
					updates[i][j][k] = 0;
				}
			}
		}
	}

	protected double error(double value, double expectedValue) {
		return expectedValue - value;
	}

	private static void accumulateUpdate(int layerIndex, double[] activatedValues, double[] localGrad,
			double[][][] updates) {
		for (int j = 0; j < localGrad.length; ++j) {
			for (int i = 0; i < activatedValues.length; ++i) {
				updates[layerIndex][i][j] += localGrad[j] * activatedValues[i];
			}
			// bias
			updates[layerIndex][activatedValues.length][j] -= localGrad[j];
		}
	}

	private static void applyActivation(double[] source, double[] results, ActivationFun activation) {
		for (int i = 0; i < source.length; ++i) {
			results[i] = activation.apply(source[i]);
		}
	}

	private static void forwardPass(BPLayer[] layers, ActivationFun[] activation, double[][] innerLayerOut,
			double[][] innerLayerOutRaw, double[] inputs) {
		layers[0].apply(inputs, innerLayerOutRaw[0]);
		applyActivation(innerLayerOutRaw[0], innerLayerOut[0], activation[0]);
		for (int i = 1; i < layers.length; i++) {
			layers[i].apply(innerLayerOut[i - 1], innerLayerOutRaw[i]);
			applyActivation(innerLayerOutRaw[i], innerLayerOut[i], activation[i]);
		}
	}

}
