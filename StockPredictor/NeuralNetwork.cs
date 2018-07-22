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
            TFTensor outputs = inputs;


            return outputs;
        }

        public void Train()
        {

        }



        #endregion


    }
}
