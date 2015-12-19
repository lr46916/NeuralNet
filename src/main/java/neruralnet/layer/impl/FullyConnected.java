package neruralnet.layer.impl;

import java.io.Serializable;
import java.util.Arrays;

import neruralnet.function.distribution.Distribution;
import neruralnet.function.distribution.impl.NormalDist;
import neruralnet.layer.Layer;
import neruralnet.layer.StatelessLayer;

public class FullyConnected extends StatelessLayer implements Serializable{

	private static final long serialVersionUID = 3878314843105195424L;
	
	private double[] weights;
	private int n, m;

	public FullyConnected(int n, int m, Distribution dist) {
		super();
		this.n = n;
		this.m = m;
		weights = dist.getElements(n * m);
	}
	
	public FullyConnected(int n, int m) {
		super();
		this.n = n;
		this.m = m;
		weights = new double[n * m];
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		assert (this.weights.length <= weights.length - offset);
		for(int i = 0; i < this.weights.length; i++) {
			this.weights[i] = weights[i + offset];
		}
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		assert (outputs.length == m && inputs.length == n);
		int c = 0;
		for (int i = 0; i < m; i++) {
			outputs[i] = 0;
			for (int j = 0; j < n; j++) {
				outputs[i] += weights[c++] * inputs[j];
			}
		}
	}

	@Override
	public double[] getWeights() {
		return weights;
	}

	@Override
	public int getNumberOfInputs() {
		return n;
	}

	@Override
	public int getNumberOfOutputs() {
		return m;
	}

	@Override
	public int getNumberOfWeights() {
		return weights.length;
	}
	
	public static void main(String[] args) {

		Layer l = new FullyConnected(3, 1, new NormalDist(0, 1));

		double[] inputs = new double[] { 0.1, 0.2, 0.3 };
		double[] outputs = new double[1];
		
		l.apply(inputs, outputs);

		System.out.println(Arrays.toString(outputs));
		
	}

}
