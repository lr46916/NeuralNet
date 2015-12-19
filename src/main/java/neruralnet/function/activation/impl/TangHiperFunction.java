package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFun;

public class TangHiperFunction implements ActivationFun, Serializable {

	private static final long serialVersionUID = -4246245062162713453L;

	@Override
	public double apply(double x) {
		double tmp = Math.pow(Math.E, -2 * x);
		return (1. - tmp) / (1. + tmp);
	}

}
