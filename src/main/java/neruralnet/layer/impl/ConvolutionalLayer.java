//package neruralnet.layer.impl;
//
//import java.io.Serializable;
//
//import neruralnet.layer.PrototypeLayer;
//import neruralnet.layer.StatelessLayer;
//
//public class ConvolutionalLayer extends StatelessLayer implements PrototypeLayer, Serializable {
//
//	private static final long serialVersionUID = 3760056306553141218L;
//
//	private int n;
//	private int m;
//	// strides might be set in the method?
//	// myb symple setters
//	private int stride;
//	
//	// myb allow constructor to define number of outputs. Then strides will be
//	// calculated based on
//	// input image dimensions and number of outputs
//	private double[] weights;
//	private int filters;
//
//	public ConvolutionalLayer(int n, int m, int stride,
//			int filters) {
//		super();
//		this.n = n;
//		this.m = m;
//		this.stride = stride;
//		this.filters = filters;
//	}
//
//	@Override
//	public void setWeigths(double[] weights, int offset) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void apply(double[] inputs, double[] outputs) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public double[] getWeights() {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public int getNumberOfInputs() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	@Override
//	public int getNumberOfOutputs() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	@Override
//	public int getNumberOfWeights() {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//
//	@Override
//	public PrototypeLayer duplicate() {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//}
