package decentralized

import (
	"fmt"
	"testing"

	libunlynx "github.com/ldsec/unlynx/lib"
	"github.com/stretchr/testify/assert"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
)

// This test runs a decentralized learning protocol with distributed neural network model.
func TestDecentralizedLearning(t *testing.T) {
	//Debug level
	log.SetDebugVisible(2)

	//Initiate the protocol configuration
	conf := new(conf)
	conf.GetConf("../config/config.json")

	//Setup the protocol
	local := onet.NewLocalTest(libunlynx.SuiTe)
	_, err := onet.GlobalProtocolRegister("NeuralNetTest", NewNeuralNetProtocol)
	assert.NoError(t, err)
	_, _, tree := local.GenTree(conf.NodeNumber, true)
	defer local.CloseAll()

	//Initiate the learning metrics
	testAccuracy := make([][]float64, conf.ExpRep)
	testF1Score := make([][][]float64, conf.ExpRep)
	testLoss := make([][]float64, conf.ExpRep)
	for i := 0; i < conf.ExpRep; i++ {
		testAccuracy[i] = make([]float64, conf.MaxIteration)
		testLoss[i] = make([]float64, conf.MaxIteration)
		testF1Score[i] = make([][]float64, conf.NClass)
		for j := 0; j < conf.NClass; j++ {
			testF1Score[i][j] = make([]float64, conf.MaxIteration)
		}
	}

	// This for loop is the main core of the decentralized protocol. It reproduce
	// the same experiment "conf.ExpRep" times. At each repetition, it records the
	// accuracies, f1 scores and loss values of the learning process.
	for i := 0; i < conf.ExpRep; i++ {
		fmt.Println("ExpRep: ", i)

		//Protocol initialization
		name := "NeuralNetTest" // Must be the same as x name
		rootInstance, err := local.CreateProtocol(name, tree)
		assert.NoError(t, err)
		protocol := rootInstance.(*NeuralNetProtocol)

		//Metric channels initialization
		testAccuracyChannel := protocol.TestAccuracyChannel
		testLossChannel := protocol.TestLossChannel
		testF1Channel := protocol.TestF1Channel

		//Starts protocol and records metrics
		protocol.Start()
		for j := 0; j < conf.MaxIteration; j++ {
			testAccuracy[i][j] = <-testAccuracyChannel
			testLoss[i][j] = <-testLossChannel
			for k := 0; k < conf.NClass; k++ {
				testF1Score[i][k][j] = <-testF1Channel
			}
		}

		close(testAccuracyChannel)
		close(testLossChannel)
		close(testF1Channel)
	}

	fmt.Println("Test accuracy: ", testAccuracy)
	fmt.Println("Test loss: ", testLoss)
	fmt.Println("Test f1: ", testF1Score)

}
