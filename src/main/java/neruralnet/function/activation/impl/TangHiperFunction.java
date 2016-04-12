package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFunBP;

public class TangHiperFunction implements ActivationFunBP, Serializable {

	private static final long serialVersionUID = -4246245062162713453L;

	@Override
	public double apply(double x) {
		return Math.tanh(x);
	}
	
	@Override
	public double gradient(double x) {
		double y = apply(x);
		return 1 - y*y;
	}

}
