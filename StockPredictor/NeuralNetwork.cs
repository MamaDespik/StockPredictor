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

            //Weights/biases from inputs to first hidden layer, then each hidden layer to the next
            foreach (int numHiddenNodes in numHiddenNodesPerLayer)
            {
                //Add the matrix that represents the weights from each node of the previous layer to each node in this layer to the list of weights
                weightRows = new List<float[]>();
                biasRows = new List<float[]>();
                for (int i = 0; i < numHiddenNodes; i++)
                {
                    value = Convert.ToSingle((Random.NextDouble()) * Scale);
                    weightRows.Add(new List<float>(Enumerable.Range(0, numPreviousLayerNodes).Select(n => value)).ToArray());
                    value = Convert.ToSingle((Random.NextDouble()) * Scale);
                    biasRows.Add(new List<float>(Enumerable.Range(0, 1).Select(n => value)).ToArray());

                }
                Weights.Add(new TFTensor(weightRows.ToArray()));
                Biases.Add(new TFTensor(biasRows.ToArray()));

                numPreviousLayerNodes = numHiddenNodes;
            }

            //Weights/Biases from last hidden layer to output
            weightRows = new List<float[]>();
            biasRows = new List<float[]>();
            for (int i = 0; i < numOutputNodes; i++)
            {
                value = Convert.ToSingle((Random.NextDouble()) * Scale);
                weightRows.Add(new List<float>(Enumerable.Range(0, numPreviousLayerNodes).Select(n => value)).ToArray());
                value = Convert.ToSingle((Random.NextDouble()) * Scale);
                biasRows.Add(new List<float>(Enumerable.Range(0, 1).Select(n => value)).ToArray());
            }
            Weights.Add(new TFTensor(weightRows.ToArray()));
            Biases.Add(new TFTensor(biasRows.ToArray()));
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
                    TFOutput layerinputs = graph.Const(previousLayerOutput); //The inputs to this layer are the outputs from the previous layer.
                    TFOutput layerweights = graph.Const(Weights[i]); //Get the weights for this layer.
                    TFOutput biases = graph.Const(Biases[i]); //Get the biases for this layer.

                    //Matrix Multiply the weights and inputs, then add the biases, then relu.
                    TFOutput inputsTimesWeights = graph.MatMul(layerweights, layerinputs);
                    TFOutput biasesAdded = graph.Add(inputsTimesWeights, biases);
                    TFOutput activation = graph.Relu(biasesAdded);

                    //Get the tensor to feed to the next layer
                    TFSession session = new TFSession(graph);
                    previousLayerOutput = session.GetRunner().Run(activation);
                }
            }


            return previousLayerOutput;
        }

        public void Train(List<TFTensor> inputs, List<TFTensor> expectedOutputs, int numEpochs)
        {
            for (int i = 0; i < numEpochs; i++)
            {
                int test = Random.Next(0, inputs.Count);
                TFTensor expectedOutput = expectedOutputs[test];
                TFTensor actualOutput = Think(inputs[test]);
                TFTensor error;
                using (TFGraph graph = new TFGraph())
                {
                    TFOutput expected = graph.Const(expectedOutput);
                    TFOutput actual = graph.Const(actualOutput);
                    TFOutput errorOutput = graph.Sub(expected, actual);
                    TFSession session = new TFSession(graph);
                    error = session.GetRunner().Run(errorOutput);
                }
            }
        }



        #endregion


    }
}
