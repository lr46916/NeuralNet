package neruralnet.function.distribution.impl;

import java.util.Random;

import neruralnet.function.distribution.Distribution;

public class NormalDist implements Distribution {

	private double m;
	private double sigma;
	private Random rand;

	public NormalDist(double m, double sigma) {
		super();
		this.m = m;
		this.sigma = sigma;
		rand = new Random();
	}

	@Override
	public double[] getElements(int n) {
		double[] res = new double[n];

		for (int i = 0; i < n; i++) {
			res[i] = m + rand.nextGaussian() * sigma;
		}

		return res;
	}

}
