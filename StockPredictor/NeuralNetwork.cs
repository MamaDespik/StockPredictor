using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using TensorFlow;

namespace StockPredictor
{
    class NeuralNetwork
    {
        #region Properties

        private List<TFTensor> Weights;
        private List<TFTensor> Biases;

        #endregion

        #region Constructors

        public NeuralNetwork(int numInputs, int numOutputs, List<int> numHiddenLayers)
        {
            Weights = new List<TFTensor>();
            Biases = new List<TFTensor>();

            long previousLayerDepth = numInputs;
            long[] dims;
            int size;
            TFTensor tempTensor;

            foreach (int layerDepth in numHiddenLayers)
            {
                dims = new[] { previousLayerDepth, layerDepth };
                size = (int)(previousLayerDepth * layerDepth * 4);
                tempTensor = new TFTensor(TFDataType.Float, dims, size);
                Weights.Add(tempTensor);

                tempTensor = new TFTensor(new List<float>(layerDepth).ToArray());
                Biases.Add(tempTensor);

                previousLayerDepth = layerDepth;
            }

            dims = new[] { previousLayerDepth, numOutputs };
            size = (int)(previousLayerDepth * numOutputs * 4);
            tempTensor = new TFTensor(TFDataType.Float, dims, size);
            Weights.Add(tempTensor);

            tempTensor = new TFTensor(new List<float>(numOutputs).ToArray());
            Biases.Add(tempTensor);

        }

        #endregion

        #region Methods

        public TFTensor Think(TFTensor inputs)
        {
            TFTensor previousLayerOutput = inputs;

            //Feed Forward through each layer in the network
            for (int i = 0; i < Weights.Count; i++)
            {
                using (TFGraph graph = new TFGraph())
                {
                    TFSession session = new TFSession(graph);

                    TFOutput layerinputs = graph.Const(previousLayerOutput); //The inputs to this layer are the outputs from the previous layer.
                    TFOutput layerweights = graph.Const(Weights[i]); //Get the weights for this layer.
                    TFOutput biases = graph.Const(Biases[i]); //Get the biases for this layer.

                    //Matrix Multiply the weights and inputs, then add the biases.
                    TFOutput layerOutput = graph.Add(graph.MatMul(layerweights, layerinputs), biases);

                    //Get the tensor to feed to the next layer
                    previousLayerOutput = session.GetRunner().Run(layerOutput);
                }
            }


            return previousLayerOutput;
        }

        public void Train()
        {

        }



        #endregion


    }
}
