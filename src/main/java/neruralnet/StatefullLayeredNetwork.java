package neruralnet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import neruralnet.function.activation.ActivationFun;
import neruralnet.layer.Layer;
import neruralnet.layer.StatefulLayer;

public class StatefullLayeredNetwork extends StatefulLayer implements Serializable {

	private static final long serialVersionUID = -5148061112964692528L;

	private StatelessLayeredNeuralNet network;
	private List<StatefulLayer> stateFullLayers;
	private int contextSize = 0;

	public StatefullLayeredNetwork(Layer[] layers, ActivationFun[] activation) {
		initalize(layers, activation);
	}

	public StatefullLayeredNetwork(List<Layer> layers, List<ActivationFun> activation) {
		Layer[] layerArray = new Layer[layers.size()];
		for (int i = 0; i < layers.size(); ++i) {
			layerArray[i] = layers.get(i);
		}
		ActivationFun[] activationArray = new ActivationFun[activation.size()];
		for (int i = 0; i < activation.size(); ++i) {
			activationArray[i] = activation.get(i);
		}
		initalize(layerArray, activationArray);
	}

	private void initalize(Layer[] layers, ActivationFun[] activation) {
		this.network = new StatelessLayeredNeuralNet(layers, activation);
		stateFullLayers = new ArrayList<>();
		for(Layer l : layers) {
			if(l.isStateful()) {
				StatefulLayer tmp = (StatefulLayer) l;
				stateFullLayers.add(tmp);
				contextSize += tmp.getContextSize();
			}
		}
	}

	public Layer[] getLayers() {
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
		double[] contextData = new double[contextSize];
		for (StatefulLayer statefullLayer : stateFullLayers) {
			for (double x : statefullLayer.getContext()) {
				contextData[c++] = x;
			}
		}
		return contextData;
	}

	@Override
	public int getContextSize() {
		return contextSize;
	}

	@Override
	public void setContext(double[] context, int offset) {
		for (StatefulLayer statefullLayer : stateFullLayers) {
			statefullLayer.setContext(context, offset);
			offset += statefullLayer.getContextSize();
		}
	}

}
