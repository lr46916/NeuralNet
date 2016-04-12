package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFunBP;

public class IdentityFunction implements ActivationFunBP, Serializable{

	private static final long serialVersionUID = 6617399491360192880L;

	@Override
	public double apply(double value) {
		return value;
	}

	@Override
	public double gradient(double x) {
		return 1;
	}

}
