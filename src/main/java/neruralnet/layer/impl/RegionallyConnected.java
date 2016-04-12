package neruralnet.layer.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import neruralnet.function.activation.ActivationFun;
import neruralnet.function.distribution.Distribution;
import neruralnet.layer.BPLayer;
import neruralnet.layer.PrototypeLayer;

public class RegionallyConnected implements PrototypeLayer, Serializable, BPLayer {

	private static final long serialVersionUID = 7809005785539123405L;

	private int[] inputShape;
	private int[] reigonSize;
	private double[] weights;
	private int numberOfInputs;
	private int numberOfOutputs;
	private int stride;

	public RegionallyConnected(int inputWidth, int inputHeight, int reigonWidht, int reigonHeight, int stride,
			Distribution dist) {
		init2D(inputWidth, inputHeight, reigonWidht, reigonHeight, stride);
		weights = dist.getElements(numberOfOutputs * (reigonHeight * reigonWidht + 1));
	}

	public RegionallyConnected(int inputWidth, int inputHeight, int reigonWidht, int reigonHeight, int stride) {
		init2D(inputWidth, inputHeight, reigonWidht, reigonHeight, stride);
		weights = new double[numberOfOutputs * (reigonHeight * reigonWidht + 1)];
	}

	public RegionallyConnected(int inputLength, int reigonLenght, int stride,
			Distribution dist) {
		init1D(inputLength, reigonLenght, stride);
		weights = dist.getElements(numberOfOutputs * reigonLenght);
	}

	public RegionallyConnected(int inputLength, int reigonLenght, int stride) {
		init1D(inputLength, reigonLenght, stride);
		weights = new double[numberOfOutputs * (reigonLenght + 1)];
	}

	private void init2D(int inputWidth, int inputHeight, int reigonWidht, int reigonHeight, int stride) {
		inputShape = new int[2];
		reigonSize = new int[2];

		inputShape[0] = inputWidth;
		inputShape[1] = inputHeight;

		reigonSize[0] = reigonWidht;
		reigonSize[1] = reigonHeight;

		numberOfInputs = inputHeight * inputWidth;
		this.stride = stride;
		// TODO calculate outputs depending on stride size should be OK
		this.numberOfOutputs = (1 + ((inputWidth - reigonWidht) / stride))
				* (1 + ((inputHeight - reigonHeight) / stride));
	}

	private void init1D(int inputLength, int reigonLenght, int stride) {
		inputShape = new int[1];
		reigonSize = new int[1];

		inputShape[0] = inputLength;
		reigonSize[0] = reigonLenght;

		numberOfInputs = inputLength;
		this.stride = stride;
		// TODO calculate outputs depending on stride size
		this.numberOfOutputs = (inputLength - reigonLenght) / stride; // should
																		// be OK
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = weights[i + offset];
		}
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {

		if (inputShape.length == 1) {
			int offset = 0;
			int weightOffset = 0;
			for (int i = 0; i < numberOfOutputs; i++) {
				outputs[i] = -weights[weightOffset++];
				for (int j = 0; j < reigonSize[0]; j++) {
					outputs[i] += weights[weightOffset++] * inputs[offset++];
				}
				offset += stride;
			}
		} else {
			int rowOffset = 0;
			int outOffset = 0;
			int weightsOffset = 0;

			while (rowOffset + reigonSize[0] <= inputShape[0]) {
				// System.out.println("Row: " + rowOffset);
				int colOffset = 0;
				while (colOffset + reigonSize[1] <= inputShape[1]) {
					// System.out.println("Col: " + colOffset);
					double sum = -weights[weightsOffset++];
					for (int i = 0; i < reigonSize[0]; i++) {
						for (int j = 0; j < reigonSize[1]; j++) {
							sum += weights[weightsOffset++] * inputs[(rowOffset + i) * inputShape[1] + colOffset + j];
						}
					}
					outputs[outOffset] = sum;
					colOffset += stride;
					outOffset++;

				}
				rowOffset += stride;
			}

		}

	}

	@Override
	public double[] getWeights() {
		return weights;
	}

	@Override
	public int getNumberOfInputs() {
		return numberOfInputs;
	}

	@Override
	public int getNumberOfOutputs() {
		return numberOfOutputs;
	}

	@Override
	public int getNumberOfWeights() {
		return weights.length;
	}

	@Override
	public boolean isStateful() {
		return false;
	}

	@Override
	public PrototypeLayer duplicate() {
		if (inputShape.length == 2) {
			return new RegionallyConnected(inputShape[0], inputShape[1], reigonSize[0], reigonSize[1], stride);
		} else {
			return new RegionallyConnected(inputShape[0], reigonSize[0], stride);
		}
	}
	
	@Override
	public double getWeight(int k, int j) {
		throw new IllegalAccessError("Not yet implemented");
	}
	
	@Override
	public void updateWeight(int i, int j, double value) {
		throw new IllegalAccessError("Not yet implemented");
	}

	@Override
	public Iterable<Integer> connectedNeuronsIndexes(int inputIndex) {
		throw new IllegalAccessError("Not yet implemented");
	}

	public static void main(String[] args) {

		RegionallyConnected regConn = new RegionallyConnected(3, 3, 2, 2, 1, (n) -> {
			double[] ret = new double[n];
			Arrays.fill(ret, 1);
			return ret;
		});

		System.out.println(regConn.getNumberOfInputs());
		System.out.println(regConn.getNumberOfOutputs());

		double[] inputs = new double[regConn.getNumberOfInputs()];
		double[] outputs = new double[regConn.getNumberOfOutputs()];

		Arrays.fill(inputs, 1);

		regConn.apply(inputs, outputs);

		System.out.println(Arrays.toString(outputs));
		System.out.println(regConn.getNumberOfWeights());
	}

}
