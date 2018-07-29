using System;
using System.CodeDom;
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
        private Random Random = new Random(0);
        private int Scale = 1;

        #endregion

        #region Constructors

        public NeuralNetwork(int numInputNodes, int numOutputNodes, List<int> numHiddenNodesPerLayer)
        {
            Weights = new List<TFTensor>();
            Biases = new List<TFTensor>();

            int numPreviousLayerNodes = numInputNodes;
            long[] matrixShape;
            int matrixDataSize;
            List<float[]> weightRows;
            List<float[]> biasRows;
            float value;
            TFTensor tempTensor;

            foreach (int numHiddenNodes in numHiddenNodesPerLayer)
            {
                //Add the matrix that represents the weights from each node of the previous layer to each node in this layer to the list of weights
                // matrixShape = new long[] { numPreviousLayerNodes, numHiddenNodes };
                // matrixDataSize = (int)(matrixShape[0] * matrixShape[1] * 4);//4 bytes per float in the matrix
                // tempTensor = new TFTensor(TFDataType.Float, matrixShape, matrixDataSize);
                // Weights.Add(tempTensor);
                weightRows = new List<float[]>();
                biasRows = new List<float[]>();
                for (int i = 0; i < numHiddenNodes; i++)
                {
                    value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
                    weightRows.Add(new List<float>(Enumerable.Range(0, numPreviousLayerNodes).Select(n => value)).ToArray());
                    value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
                    biasRows.Add(new List<float>(Enumerable.Range(0, 1).Select(n => value)).ToArray());
                    
                }
                Weights.Add(new TFTensor(weightRows.ToArray()));
                Biases.Add(new TFTensor(biasRows.ToArray()));

                //Add the 1D array that represents the biases for each node in this layer to the list of biases
                //value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
                //List<float> test = new List<float>(Enumerable.Range(0, numHiddenNodes).Select(n => value));
                //Biases.Add(new TFTensor(test.ToArray()));

                numPreviousLayerNodes = numHiddenNodes;
            }

            //Add the matrix that represents the weights from each input to each node in this layer to the list of weights
            //matrixShape = new long[] { numPreviousLayerNodes, numOutputNodes };
            //matrixDataSize = (numPreviousLayerNodes * numOutputNodes * 4);
            //tempTensor = new TFTensor(TFDataType.Float, matrixShape, matrixDataSize);
            //Weights.Add(tempTensor);
            weightRows = new List<float[]>();
            biasRows = new List<float[]>();
            for (int i = 0; i < numOutputNodes; i++)
            {
                value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
                weightRows.Add(new List<float>(Enumerable.Range(0, numPreviousLayerNodes).Select(n => value)).ToArray());
                value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
                biasRows.Add(new List<float>(Enumerable.Range(0, 1).Select(n => value)).ToArray());
            }
            Weights.Add(new TFTensor(weightRows.ToArray()));
            Biases.Add(new TFTensor(biasRows.ToArray()));

            //Add the 1D array that represents the biases for each node in this layer to the list of biases
            //value = Convert.ToSingle((Random.NextDouble() - .5) * Scale);
            //Biases.Add(new TFTensor(new List<float>(Enumerable.Range(0, numOutputNodes).Select(n => value)).ToArray()));
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
                    TFOutput product = graph.MatMul(layerweights, layerinputs);
                    TFOutput layerOutput = graph.Add(product, biases);

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
