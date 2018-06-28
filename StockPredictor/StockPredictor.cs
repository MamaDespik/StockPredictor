using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static AlphaVantageApiWrapper.AlphaVantageApiWrapper;
using static VantageKey.VantageKey;

namespace StockPredictor
{
    class StockPredictor
    {
        const string VANTAGE_KEY = VantageKey.VantageKey.ValueType;
        static void Main(string[] args)
        {
            var parameters = new List<ApiParam>();
            parameters.Add(new ApiParam("function", "TIME_SERIES_DAILY"));
            parameters.Add(new ApiParam("symbol", "MSFT"));
            parameters.Add(new ApiParam("outputsize", "full"));

            Console.WriteLine("Getting Data...");
            AlphaVantageRootObject result = GetTechnical(parameters, VANTAGE_KEY).Result;
            Console.WriteLine("Data received: " + result.TechnicalsByDate.Count);

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

            List<List<double>> groupings = new List<List<double>>();
            int groupSize = 14;

            for (int i = groupSize; i < allValues.Count; i++)
            {
                List<double> tempGroup= new List<double>();
                for (int j = i-groupSize; j < i; j++)
                {
                    tempGroup.Add(allValues[j]);
                }
                groupings.Add(tempGroup);
            }


            while (true)
            {
                if (Console.ReadLine() == "exit") return;
                Console.WriteLine("waiting...");

            }

        }
    }
}
