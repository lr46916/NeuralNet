package neruralnet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import neruralnet.layer.Layer;
import neruralnet.layer.StatefulLayer;

public class StatefullLayeredNetwork extends StatefulLayer implements Serializable {

	private static final long serialVersionUID = -5148061112964692528L;

	private StatelessLayerdNeuralNet network;
	private List<StatefulLayer> stateFullLayers;
	private double[] contextData;
	
	public StatefullLayeredNetwork(Layer... layers) {
		super();
		this.network = new StatelessLayerdNeuralNet(layers);
		stateFullLayers = new ArrayList<>();
		int contextSize = 0;
		for(Layer l : layers) {
			if(l.isStateful()) {
				StatefulLayer tmp = (StatefulLayer) l;
				stateFullLayers.add(tmp);
				contextSize += tmp.getContextSize();
			}
		}
		contextData = new double[contextSize];
	}
	
	public Layer[] getLayers(){
		return network.getLayers();
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		network.setWeigths(weights, offset);
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		network.apply(inputs, outputs);
	}

	@Override
	public double[] getWeights() {
		return network.getWeights();
	}

	@Override
	public int getNumberOfInputs() {
		return network.getNumberOfInputs();
	}

	@Override
	public int getNumberOfOutputs() {
		return network.getNumberOfOutputs();
	}

	@Override
	public int getNumberOfWeights() {
		return network.getNumberOfWeights();
	}

	@Override
	public double[] getContext() {
		int c = 0;
		for(StatefulLayer statefullLayer : stateFullLayers) {
			for(double x : statefullLayer.getContext()) {
				contextData[c++] = x;
			}
		}
		return contextData;
	}

	@Override
	public int getContextSize() {
		return contextData.length;
	}

	@Override
	public void setContext(double[] context, int offset) {
		for(StatefulLayer statefullLayer : stateFullLayers) {
			statefullLayer.setContext(context, offset);
			offset += statefullLayer.getContextSize();
		}
	}

}
