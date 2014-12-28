using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NN_Project
{

    class NeuralNetwork
    {
        //ANN Parameters (aActParam and bActParam are sigmoid and tanh parameters) 
        double ETA, aActParam, bActParam, maxIters, maxError;
        char actFun;                                                    //actFun = 'S' -> Sigmoid .. actFun = 'T' -> Tanh
        List<double> errorList;                                         //Holds all error percentages for each epoch
        List<int> nnLayers;                                             //Holds number of neuron for each layer
        List<List<List<double>>> weights;                               //Holds all the weights .. Note : same lecture's indexing
                                                                        //weights[layer][rightNeuron][leftNeuron]
        List<List<double>> activationValue, deltaValue;                 //Hold propagations' values
        double maxAbsOfData = 0;
        List<double> mean;

        public NeuralNetwork(List<int> _nnLayers, double _ETA, double _maxError,
            double _maxIters, char _actFun, double _aActParam, double _bActParam)         //MaxError : maximum allowed error
        {
            actFun = _actFun;
            maxError = _maxError;
            maxIters = _maxIters;
            aActParam = _aActParam;
            bActParam = _bActParam;
            ETA = _ETA;
            nnLayers = new List<int>(_nnLayers);
            errorList = new List<double>();
            activationValue = new List<List<double>>();
            deltaValue = new List<List<double>>();
            weights = new List<List<List<double>>>();
            for (int i = 0; i < nnLayers.Count; ++i)
                nnLayers[i]++;
            Random rand = new Random();                         //Random weights initialization
            for (int i = 0; i < nnLayers.Count - 1; ++i)
            {
                weights.Add(new List<List<double>>());
                for (int j = 0; j < nnLayers[i + 1]; ++j)
                {
                    weights[i].Add(new List<double>());
                    for (int k = 0; k < nnLayers[i]; ++k)
                        weights[i][j].Add(rand.NextDouble()/1000.0);
                }
            }
            nnLayers.RemoveAt(0); //Remove the number of input neurons .. No need for it
        }
        private double sigmoid(double val)
        {
            return 1 / (1 + Math.Exp(-aActParam * val));
        }
        private double tanh(double val)
        {
            return aActParam * Math.Tanh(bActParam * val);
        }
        private double activationFunction(char func, bool derivative, double val)        //When derivative=true, 
        {                                                                                //it returns the derivatives of z value
            if (!derivative)
                return func == 'S' ? sigmoid(val) : tanh(val);
            else if (func == 'S')
                return aActParam * (val) * (1 - (val));
            else
                return (bActParam / aActParam) * (aActParam - (val)) * (aActParam + (val));
        }
        private List<double> getMean(List<List<double>> Samples)
        {
            List<double> Mean = new List<double>();

            for (int x = 0; x < Samples[0].Count; ++x)
            {
                double sum = 0.0;
                for (int i = 0; i < Samples.Count; ++i)
                    sum += Samples[i][x];
                Mean.Add(sum / (double)Samples.Count);
            }

            return Mean;
        }

        private void updateNormalizationParam(List<List<double>> data)
        {
            mean = getMean(data);
            for (int i = 0; i < data.Count; ++i)
                for (int j = 0; j < data[i].Count; ++j)
                    maxAbsOfData = Math.Max(maxAbsOfData, Math.Abs(data[i][j] - mean[j]));
        }
        private void normalize(List<List<double>> data)
        {
            for (int i = 0; i < data.Count; ++i)
                for (int j = 0; j < data[i].Count; ++j)
                {
                    data[i][j] -= mean[j];
                    data[i][j] /= maxAbsOfData;
                }
        }
        private List<List<double>> copyList(List<List<double>> l)
        {
            List<List<double>> ll = new List<List<double>>();
            for (int i = 0; i < l.Count; ++i)
            {
                ll.Add(new List<double>());
                for (int j = 0; j < l[i].Count; ++j)
                    ll[i].Add(l[i][j]);
            }
            return ll;
        }
        private void backwardPropagation(int trainingLabel)                             //trainingLabel needed to 
        {                                                                               //compute output neurons' deltas.

            for (int l = nnLayers.Count - 1; l >= 0; --l)                               //Looping backward from the output to 
            {                                                                           //the first hidden layer.
                List<double> prevLayer, curLayer = deltaValue[l];                       //prevLayer holds the right layer 
                //while curLayer holds the left layer
                for (int i = 1; i < curLayer.Count; ++i)
                {
                    double actValueDerivative = activationFunction(actFun, true, activationValue[l][i]);
                    if (l == nnLayers.Count - 1)                                                                  //In the output layer
                        deltaValue[l][i] = ((trainingLabel == i ? 1 : actFun=='S'?0:-1) - activationValue[l][i]) * actValueDerivative; //If the current neuron 
                    else                                                                                          //should be one, put 1 else 0
                    {
                        prevLayer = deltaValue[l + 1];

                        curLayer[i] = 0;
                        for (int j = 1; j < prevLayer.Count; ++j)
                            curLayer[i] += prevLayer[j] * weights[l + 1][j][i] * actValueDerivative;              //Calculating delta for the
                        //non-output neurons
                    }
                }
            }
        }
        private void updateWeights(List<double> trainingData)
        {
            for (int l = 0; l < nnLayers.Count; ++l)                                               //For each weight -> Update
                for (int j = 1; j < weights[l].Count; ++j)
                    for (int i = 0; i < weights[l][j].Count; ++i)
                    {
                        List<double> prevLayer;                                                    //Holds left layer's activation values
                        if (l == 0)                                                                //layer = 0 .. weights between the 
                            prevLayer = trainingData;                                              //trainingData and the first hidden layer
                        else
                            prevLayer = activationValue[l - 1];
                        weights[l][j][i] += ETA * deltaValue[l][j] * prevLayer[i];
                    }
        }
        private List<double> forwardPropagation(List<double> trainingData)
        {
            List<double> output = new List<double>();
            for (int l = 0; l < nnLayers.Count; ++l)                                    //Looping forward from the first hidden layer to
            {                                                                           //the input.
                List<double> prevLayer, curLayer = activationValue[l];                  //prevLayer holds the left layer 
                                                                                        //while curLayer holds the right layer
                if (l == 0)
                    prevLayer = trainingData;
                else
                    prevLayer = activationValue[l - 1];
                for (int j = 1; j < curLayer.Count; ++j)
                {
                    curLayer[j] = 0;
                    for (int i = 0; i < prevLayer.Count; ++i)
                        curLayer[j] += prevLayer[i] * weights[l][j][i];                 //Calculating the activation value foreach neuron
                    curLayer[j] = activationFunction(actFun, false, curLayer[j]);
                    if (l == nnLayers.Count - 1)                                        //Copying the output activation value (the result)
                        output.Add(curLayer[j]);
                }
            }
            return output;
        }
        public int classify(List<double> patternFeatures)
        {
            patternFeatures.Insert(0, 1);
            List<double> output = forwardPropagation(patternFeatures);
            return output.IndexOf(output.Max()) + 1;
        }

        public double startTestingMSE(List<List<double>> TestingData, List<int> TestingLabels)
        {
            TestingData = copyList(TestingData);
            normalize(TestingData);
            for (int t = 0; t < TestingData.Count; ++t)                                //Adding 1 for each patterm (the bias term)
                TestingData[t].Insert(0, 1);
            double sumPatternError = 0;
            sumPatternError = 0;
            for (int t = 0; t < TestingData.Count; ++t)                           //Lists Initialization
            {
                List<double> lastLayerOutput = forwardPropagation(TestingData[t]);
                for (int i = 0; i < lastLayerOutput.Count; ++i)
                    sumPatternError += (lastLayerOutput[i] - ((TestingLabels[t] - 1) == i ? 1 : 0)) *
                        (lastLayerOutput[i] - ((TestingLabels[t] - 1) == i ? 1 : 0));
                
            }
            double ErrorPercentage = sumPatternError / (double)(4 * TestingData.Count);

            return ErrorPercentage;
        }
        public double startTesting(List<List<double>> TestingData, List<int> TestingLabels)
        {
            TestingData = copyList(TestingData);
            normalize(TestingData);
            for (int t = 0; t < TestingData.Count; ++t)                                //Adding 1 for each patterm (the bias term)
                TestingData[t].Insert(0, 1);
            double sumPatternError = 0;
            sumPatternError = 0;
            for (int t = 0; t < TestingData.Count; ++t)                           //Lists Initialization
            {
                List<double> lastLayerOutput = forwardPropagation(TestingData[t]);
                sumPatternError += lastLayerOutput.IndexOf(lastLayerOutput.Max()) == (TestingLabels[t] - 1)?0:1;
            }
            double ErrorPercentage = sumPatternError / (double)TestingData.Count;

            return ErrorPercentage;
        }
        public double startTraining(List<List<double>> trainingData, List<int> trainingLabels)
        {
            trainingData = copyList(trainingData);
            updateNormalizationParam(trainingData);
            normalize(trainingData);
            for (int t = 0; t < trainingData.Count; ++t)                                //Adding 1 for each patterm (the bias term)
                trainingData[t].Insert(0, 1);
            double sumPatternError = 0;
            for (int epoch = 0; epoch < maxIters; ++epoch)
            {
                sumPatternError = 0;
                for (int t = 0; t < trainingData.Count; ++t)                           //Lists Initialization
                {
                    activationValue.Clear(); deltaValue.Clear();
                    for (int i = 0; i < nnLayers.Count; ++i)
                    {
                        double[] nnLayerList = new double[nnLayers[i]]; nnLayerList.Fill(1);
                        activationValue.Add(new List<double>(nnLayerList));
                        deltaValue.Add(new List<double>(nnLayerList));
                    }
                   
                    //Forward Propagation
                    List<double> lastLayerOutput = forwardPropagation(trainingData[t]);
                    for (int i = 0; i < lastLayerOutput.Count; ++i)
                        sumPatternError += (lastLayerOutput[i] - ((trainingLabels[t] - 1) == i ? 1 : 0)) *
                            (lastLayerOutput[i] - ((trainingLabels[t] - 1) == i ? 1 : 0));

                    //Backward Propagation
                    backwardPropagation(trainingLabels[t]);

                    //Updating the weights
                    updateWeights(trainingData[t]);

                }
                errorList.Add((sumPatternError / (double)trainingData.Count));
                if (errorList.Last() < maxError)
                    break;
            }
            return errorList.Last();
        }
    }

}

public static class ArrayExtensions
{
    public static void Fill<T>(this T[] originalArray, T with)
    {
        for (int i = 0; i < originalArray.Length; i++)
        {
            originalArray[i] = with;
        }
    }
}
