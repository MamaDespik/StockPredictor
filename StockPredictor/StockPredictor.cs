using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using static AlphaVantageApiWrapper.AlphaVantageApiWrapper;
using static VantageKey.VantageKey;
using System.Numerics;

namespace StockPredictor
{
    class StockPredictor
    {
        const string VANTAGE_KEY = VantageKey.VantageKey.ValueType;
        static void Main(string[] args)
        {

            #region Fetch Data
            List<ApiParam> parameters = new List<ApiParam>();
            parameters.Add(new ApiParam("function", "TIME_SERIES_DAILY"));
            parameters.Add(new ApiParam("symbol", "MSFT"));
            parameters.Add(new ApiParam("outputsize", "full"));

            Console.WriteLine("Getting Data...");
            AlphaVantageRootObject result = GetTechnical(parameters, VANTAGE_KEY).Result;
            Console.WriteLine("Data received: " + result.TechnicalsByDate.Count);
            #endregion

            #region Prepare Data
            List<double> allValues = new List<double>();

            foreach (TechnicalDataDate dateAndData in result.TechnicalsByDate)
            {
                double avg = (dateAndData.Data[1].TechnicalValue + dateAndData.Data[2].TechnicalValue) / 2;
                Console.WriteLine(dateAndData.Date + ": " + avg);
                allValues.Add(avg);
            }
            Console.WriteLine();
            Console.WriteLine("Done printing " + result.TechnicalsByDate.Count + " datapoints.");
            Console.WriteLine();

            //Create a set of data that contains two weeks of previous stock values for each day
            List<List<double>> dataSets = new List<List<double>>();
            int groupSize = 10;
            for (int i = groupSize; i < allValues.Count; i++)
            {
                List<double> tempGroup = new List<double>();
                for (int j = i - groupSize; j < i; j++)
                {
                    tempGroup.Add(allValues[j]);
                }
                dataSets.Add(tempGroup);
            }

            //Split the datasets into training and testing data
            List<List<double>> trainingData = new List<List<double>>();
            List<List<double>> testingData = new List<List<double>>();
            double dataRatio = .8;
            int splitIndex = (int)(dataRatio * dataSets.Count);

            for (int i = 0; i < splitIndex; i++) trainingData.Add(dataSets[i]);
            for (int i = splitIndex; i < dataSets.Count; i++) testingData.Add(dataSets[i]);

            //Split the data into inputs and expectedOutputs
            List<List<double>> trainingInputs = new List<List<double>>();
            List<List<double>> trainingExpectedOutputs = new List<List<double>>();
            List<List<double>> testingInputs = new List<List<double>>();
            List<List<double>> testingExpectedOutputs = new List<List<double>>();
            foreach (List<double> dataset in trainingData)
            {
                trainingInputs.Add(dataset.GetRange(0, groupSize - 1));
                //If today's value is less than tomorrows, buy don't sell.  Otherwise, don't buy sell
                List<double> expectedOutput = dataset[groupSize - 2] < dataset[groupSize - 1] ?
                    new List<double> { 1, 0 } : new List<double> { 0, 1 };
                trainingExpectedOutputs.Add(expectedOutput);
            }
            foreach (List<double> dataset in testingData)
            {
                testingInputs.Add(dataset.GetRange(0, groupSize - 1));
                //If today's value is less than tomorrows, buy don't sell.  Otherwise, don't buy sell
                List<double> expectedOutput = dataset[groupSize - 2] < dataset[groupSize - 1] ? 
                    new List<double> { 1, 0 } : new List<double> { 0, 1 };
                testingExpectedOutputs.Add(expectedOutput);
            }

            //create the tensors
            TFTensor trainingInputsTensor = GetTensorFromData(trainingInputs);
            TFTensor trainingExpectedOutputsTensor = GetTensorFromData(trainingExpectedOutputs);
            TFTensor testingInputsTensor = GetTensorFromData(testingInputs);
            TFTensor testingExpectedOutputsTensor = GetTensorFromData(testingExpectedOutputs);


            #endregion

            #region Create NN

            TFGraph graph = new TFGraph();

            TFSession session = new TFSession(graph);

            //session.GetRunner().AddInput() //Bad


            #endregion

            while (true)
            {
                if (Console.ReadLine() == "exit") return;
                Console.WriteLine("waiting...");

            }

        }

        public static TFTensor GetTensorFromData(List<List<double>> data)
        {
            List<double[]> tempList = new List<double[]>();
            foreach (List<double> subSet in data)tempList.Add(subSet.ToArray());
            double[][] arrayData = tempList.ToArray();
            return new TFTensor(arrayData);
        }
    }
}
