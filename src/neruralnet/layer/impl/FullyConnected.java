package neruralnet.layer.impl;

import neruralnet.layer.Layer;

public class FullyConnected implements Layer {

	private double[] weigths;

	public FullyConnected(double[] weigths) {
		super();
		this.weigths = weigths;
	}
	
	@Override
	public void setWeigths(double[] weights) {
		this.weigths = weights;
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		
	}

	@Override
	public double[] getWeights() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getNumberOfInputs() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getNumberOfOutputs() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getNumberOfWeights() {
		// TODO Auto-generated method stub
		return 0;
	}

}
