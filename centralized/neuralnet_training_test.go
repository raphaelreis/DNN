package centralized

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	libspindle "github.com/ldsec/dpnn/lib"
	"github.com/ldsec/dpnn/protocols/benchmarking/neural_network/dataloader"
	"github.com/ldsec/dpnn/protocols/benchmarking/neural_network/stats"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func TestCentralizedLearning(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	//Load configuration file
	var conf conf
	conf.GetConf("../config/config.json")

	// Load data
	train, err := dataloader.Load(conf.Dataset, conf.TrainSet)
	if err != nil {
		panic(err)
	}
	test, err := dataloader.Load(conf.Dataset, conf.TestSet)
	if err != nil {
		panic(err)
	}
	test.Shuffle()
	train.Shuffle()
	tr, val := train.Split(0.8)
	fmt.Println("train: ", train[0])
	fmt.Println("Dataset is loaded.")

	// Metric initiation
	testAccuracy := make([][]float64, conf.ExpRep)
	testF1Score := make([][][]float64, conf.ExpRep)
	testLoss := make([][]float64, conf.ExpRep)
	for i := 0; i < conf.ExpRep; i++ {
		testAccuracy[i] = make([]float64, conf.Epochs)
		testLoss[i] = make([]float64, conf.Epochs)
		testF1Score[i] = make([][]float64, conf.NClass)
		for j := 0; j < conf.NClass; j++ {
			testF1Score[i][j] = make([]float64, conf.Epochs)
		}
	}

	// Activation function parser
	var activation deep.ActivationType
	if conf.Activation == "Sigmoid" {
		if conf.ActivationType == "Continuous" {
			activation = deep.ActivationSigmoid
		} else {
			activation = deep.ActivationSigmoidApproxClear
		}
	}

	// This for loop is the main core of the centralized protocol. It reproduce
	// the same experiment "conf.ExpRep" times. At each repetition, it records the
	// accuracies, f1 scores and loss values of the learning process.
	for i := 0; i < conf.ExpRep; i++ {
		fmt.Println("ExpRep: ", i)

		// Initialization of the model
		neural := deep.NewNeural(&deep.Config{
			//Size of the input
			Inputs: len(train[0].Input),
			//Architecture of the model
			Layout: conf.Layout,
			//Activation function for the hidden layers
			Activation: activation,
			//Type of prediction (MultiClass, Binary, Regression, MultiLabel)
			Mode: deep.ModeMultiClass,
			//Method for initialization (Uniform, Normal, KaimingHe)
			Weight: deep.NewKaimingHe(-1., 1.),
			Bias:   true,
		})

		for j := 0; j < conf.Epochs; j++ {
			fmt.Println("Epoch", j)

			// Training
			trainer := training.NewBatchTrainer(training.NewSGD(conf.LearningRate, conf.Momentum, conf.Decay, conf.Nesterov), conf.VerboseTrain, conf.BatchSize, 8)
			trainer.Train(neural, tr, val, 1)

			// Analysis
			testPredicted, testActual := stats.PredictionVsActual(neural, test)
			testAccuracy[i][j] = libspindle.AccuracyMultinomial(testPredicted, testActual)
			for k := 0; k < conf.NClass; k++ {
				testF1Score[i][k][j] += libspindle.FscoreMultinomial(int64(k), testPredicted, testActual)
			}
			testLoss[i][j] = stats.Loss(neural, test)
			// Print the running accuracy
			fmt.Println("test accuracy: ", libspindle.AccuracyMultinomial(testPredicted, testActual))
		}
	}

	fmt.Println("testAccuracy: ", testAccuracy)
	fmt.Println("testF1Score: ", testF1Score)
	fmt.Println("testLoss: ", testLoss)
}
