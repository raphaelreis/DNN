package decentralized

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"runtime"

	libspindle "github.com/ldsec/dpnn/lib"
	"github.com/ldsec/dpnn/protocols/benchmarking/neural_network/dataloader"
	"github.com/ldsec/dpnn/protocols/benchmarking/neural_network/stats"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
)

const NeuralNetProtocolName = "NeuralNetProtocol"

func init() {
	network.RegisterMessage(DataAccessMessage{})
	network.RegisterMessage(ChildUpdatedLocalWeightsMessage{})
	network.RegisterMessage(NewIterationMessage{})
	network.RegisterMessage(DataAccessMessage{})
	if _, err := onet.GlobalProtocolRegister(NeuralNetProtocolName, NewNeuralNetProtocol); err != nil {
		log.Fatal("Failed to register the <BatchGradProtocol>")
	}
}

type NeuralNetProtocol struct {
	*onet.TreeNodeInstance

	// Protocol configuration
	ProtocolConf conf

	// Metric channels
	TestAccuracyChannel chan float64
	TestF1Channel       chan float64
	TestLossChannel     chan float64

	// Root Channel
	WaitChannel chan int

	// Protocol communication channels
	AnnouncementChannel      chan newIterationAnnouncementStruct
	ChildLocalWeightsChannel chan []ChildUpdatedLocalWeightsMessageStruct
	DataAccessChannel        chan DataAccessStruct

	Config        deep.Config
	LocalWeights  [][][]float64
	GlobalWeights [][][]float64

	MaxIterations   int
	IterationNumber int

	// Failure handling
	FCounter int
	Dead     bool
}

// Check that *TemplateProtocol implements onet.ProtocolInstance
var _ onet.ProtocolInstance = (*NeuralNetProtocol)(nil)

// NewNeuralNetProtocol initialises the structure for use in one round
func NewNeuralNetProtocol(n *onet.TreeNodeInstance) (onet.ProtocolInstance, error) {

	//Configuration file initialization
	conf := new(conf)
	conf.GetConf("../config/config.json")

	//Protocol initialization
	pap := &NeuralNetProtocol{
		TreeNodeInstance:    n,
		ProtocolConf:        *conf,
		TestAccuracyChannel: make(chan float64),
		TestF1Channel:       make(chan float64),
		TestLossChannel:     make(chan float64),
		WaitChannel:         make(chan int),
		MaxIterations:       conf.MaxIteration,
		IterationNumber:     0,
		FCounter:            0,
	}

	// Neural network activation function parsing
	var activation deep.ActivationType
	if conf.Activation == "Sigmoid" {
		if conf.ActivationType == "Continuous" {
			activation = deep.ActivationSigmoid
		} else {
			activation = deep.ActivationSigmoidApproxClear
		}
	} else {
		activation = deep.ActivationReLU
	}

	// Initialization of the model
	config := deep.Config{
		//Size of the input
		Inputs: conf.InputSize,
		//Architecture of the model
		Layout: conf.Layout,
		//Activation function for the hidden layers
		Activation: activation,
		//Type of prediction (MultiClass, Binary, Regression, MultiLabel)
		Mode: deep.ModeBinary,
		//Method for initialization (Uniform, Normal, KaimingHe)
		Weight: deep.NewKaimingHe(-1, 1), // slight positive bias helps ReLU
		Bias:   true,
	}

	pap.Config = config

	// Registration of the communication channels
	err := pap.RegisterChannel(&pap.DataAccessChannel)
	if err != nil {
		return nil, errors.New("couldn't register data access channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.AnnouncementChannel)
	if err != nil {
		return nil, errors.New("couldn't register announcement channel: " + err.Error())
	}

	err = pap.RegisterChannel(&pap.ChildLocalWeightsChannel)
	if err != nil {
		return nil, errors.New("couldn't register child-error channel: " + err.Error())
	}

	return pap, nil
}

// Start sends the Announce-message to all children
func (p *NeuralNetProtocol) Start() error {

	// Data Access initilization
	// When working on a local protocol, we build a map for each node to access
	// the their training data. This map has the "serverId" as a key and the value
	// is a chunk of the data.
	dataAccessMessage := p.buildDataAccessMap()
	if err := p.SendToChildren(&dataAccessMessage); err != nil {
		log.Fatal("Error sending <dataAccessMessage>: ", p.ServerIdentity().String(), " :", err)
	}

	// Model initialization
	net := deep.NewNeural(&p.Config)
	globalWeights := net.Weights()
	p.GlobalWeights = globalWeights
	p.LocalWeights = globalWeights

	// Builds a byte version of the model parameters
	globalWeightsToSend, err := marshalWeights(p.GlobalWeights)
	if err != nil {
		log.Fatal("json marshalling failed with error: ", err)
	}

	// Builds a new iteration message that contains the iteration number 0 since
	// this is the beginning of the protocol. It also sends the global model
	// parameters.
	newIterationMessage := NewIterationMessage{p.IterationNumber, globalWeightsToSend}

	// Unlock channel (root node can continue with Dispatch)
	p.WaitChannel <- 1

	// Sends the new iteration message to the children
	if err := p.SendToChildren(&newIterationMessage); err != nil {
		log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String())
	}
	return nil
}

// Dispatch implements the main logic of the protocol. The function is only
// called once. The protocol is considered finished when Dispatch returns and
// Done is called.
func (p *NeuralNetProtocol) Dispatch() error {
	defer p.Done()

	// Wait for the initialization of the weights (this is done in the Start())
	if p.IsRoot() {
		<-p.WaitChannel
	}

	// Each server except the the root server is randomly selected to go into failure
	// mode.
	if !p.IsRoot() {
		if rand.Float64() <= p.ProtocolConf.FailureRate {
			p.Dead = true
		} else {
			p.Dead = false
		}
	}

	// Get the data from the parent message
	var train, val, test training.Examples
	if !p.IsRoot() {
		train, val = p.receiveData()
	} else {
		test, _ = p.receiveData()
	}

	newIterationMessage := NewIterationMessage{}
	aggregatedWeights := p.LocalWeights

	// Initialize the local model
	net := deep.NewNeural(&p.Config)

	// Protocol iterations
	for p.IterationNumber < p.MaxIterations {
		// Counter of failing servers
		p.FCounter = 0

		if p.IsRoot() {
			fmt.Println("iteration: ", p.IterationNumber)
		}

		// 1. Forward Pass announcement phase
		finished := false
		if !p.IsRoot() {

			// The root catch the parental newIterationMessage
			newIterationMessage = p.newIterationAnnouncementPhase()
			p.IterationNumber = newIterationMessage.IterationNumber
			err := unmarshalWeights(newIterationMessage.GlobalWeights, &p.GlobalWeights)
			if err != nil {
				log.Fatal("Unmarshalling failed with error: ", err)
			}
			err = unmarshalWeights(newIterationMessage.GlobalWeights, &p.LocalWeights)
			if err != nil {
				log.Fatal("Unmarshalling failed with error: ", err)
			}
			finished = p.IterationNumber >= p.MaxIterations //need to check as new number is part of the message for non root nodes
		}

		if !finished {

			// For all nodes different from root and not in dead mode, there is
			// a training phase on local data
			if !p.IsRoot() && !p.Dead {
				// Training
				net.ApplyWeights(p.GlobalWeights)
				trainer := training.NewBatchTrainer(training.NewSGD(p.ProtocolConf.LearningRate, p.ProtocolConf.Momentum, p.ProtocolConf.Decay, p.ProtocolConf.Nesterov), p.ProtocolConf.VerboseTrain, p.ProtocolConf.BatchSize, 8)
				trainer.Train(net, train, val, p.ProtocolConf.Epochs)

				p.LocalWeights = ponderateParameters(net.Weights(), len(train))
			}

			aggregatedWeights = p.LocalWeights

			// If the node is not a leaf, it aggregate its children weights
			if !p.IsLeaf() {
				// If the node is not dead it will add up all the children weights
				// with its weights. If the node is dead, it will add up with a
				// drained version of the model's parameters.
				if !p.Dead {
					for _, v := range <-p.ChildLocalWeightsChannel {

						buf := v.ChildLocalWeights
						var w [][][]float64
						err := unmarshalWeights(buf, &w)
						if err != nil {
							log.Fatal("Unmarshalling failed with error: ", err)
						}
						p.FCounter += v.FCounter
						aggregatedWeights = add3d(aggregatedWeights, w)
					}
				} else {
					drain(&aggregatedWeights)
					for _, v := range <-p.ChildLocalWeightsChannel {

						buf := v.ChildLocalWeights
						var w [][][]float64
						err := unmarshalWeights(buf, &w)
						if err != nil {
							log.Fatal("Unmarshalling failed with error: ", err)
						}

						p.FCounter += v.FCounter
						aggregatedWeights = add3d(aggregatedWeights, w)
					}
				}

			}

			// If the node is not the root, it sends its aggregated weight to the parent
			if !p.IsRoot() {
				aggregatedWeightsToSend, err := marshalWeights(aggregatedWeights)

				if err != nil {
					log.Fatal(p.ServerIdentity().Address, "json marshalling failed with error: ", err)
				}
				if p.Dead {
					log.Lvl3(p.ServerIdentity().Address, "sends to", p.Parent().ServerIdentity.Address)
					if err := p.SendToParent(&ChildUpdatedLocalWeightsMessage{aggregatedWeightsToSend, p.FCounter + 1}); err != nil {
						log.Fatal("Error sending <ChildErrorBytesMessage>: ", p.ServerIdentity().String())
					}
				} else {
					log.Lvl3(p.ServerIdentity().Address, "sends to", p.Parent().ServerIdentity.Address)
					if err := p.SendToParent(&ChildUpdatedLocalWeightsMessage{aggregatedWeightsToSend, p.FCounter}); err != nil {
						log.Fatal("Error sending <ChildErrorBytesMessage>: ", p.ServerIdentity().String())
					}
				}
			}

			if p.IsRoot() {
				gobalWeightsToSend := p.buildGlobalWeights(net, aggregatedWeights, p.ProtocolConf.TrainSize)
				p.sendMetrics(net, train, test)
				p.IterationNumber = p.IterationNumber + 1
				newIterationMessage := NewIterationMessage{p.IterationNumber, gobalWeightsToSend}
				if err := p.SendToChildren(&newIterationMessage); err != nil {
					log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String())
				}
			}
		}
		if p.IsRoot() {
			fmt.Println("Number of dead nodes: ", p.FCounter)
			fmt.Printf("#goroutines: %d\n", runtime.NumGoroutine())
		}
	}
	return nil
}

//sendMetrics send the metrics from the master node to the test instance.
func (p *NeuralNetProtocol) sendMetrics(net *deep.Neural, train training.Examples, test training.Examples) {
	testPredicted, testActual := stats.PredictionVsActual(net, test)

	testAccuracy := libspindle.AccuracyMultinomial(testPredicted, testActual)
	testLoss := stats.Loss(net, test)

	p.TestAccuracyChannel <- testAccuracy
	p.TestLossChannel <- testLoss

	for i := 0; i < p.ProtocolConf.NClass; i++ {
		testF1Score := libspindle.FscoreMultinomial(int64(i), testPredicted, testActual)
		p.TestF1Channel <- testF1Score
	}
}

// buildDataAccessMap set up the training and test data set map for local protocol.
func (p *NeuralNetProtocol) buildDataAccessMap() DataAccessMessage {

	// Load the training data
	train, err := dataloader.Load(p.ProtocolConf.Dataset, p.ProtocolConf.TrainSet)
	if err != nil {
		panic(err)
	}
	train.Shuffle()
	p.ProtocolConf.TrainSize = float64(len(train))

	// Split the training data in the number of nodes
	batch := len(train) / (p.ProtocolConf.NodeNumber - 1)
	log.Lvl3("Chuck size: ", batch)
	nonOverlappingTrain := train.SplitSize(batch)
	log.Lvl3("Training chunks: ", len(nonOverlappingTrain))

	// Build the map for workers to access their data
	treeNodes := p.List()
	accessMap := make(map[string]*training.Examples, p.ProtocolConf.NodeNumber-1)
	for _, node := range treeNodes {
		chunck := 0
		if node.ID != p.Root().ID {
			accessMap[node.ID.String()] = &nonOverlappingTrain[chunck]
			chunck++
		}
	}
	dataAccessMapToSend, err := json.Marshal(accessMap)
	if err != nil {
		log.Fatal("json marshalling failed with error: ", err)
	}

	return DataAccessMessage{dataAccessMapToSend}
}

//receiveData return the train and validation set if the node is a worker and test set if
// the node is the master.
func (p *NeuralNetProtocol) receiveData() (training.Examples, training.Examples) {
	var train, val training.Examples
	if !p.IsRoot() {
		// Get access to the data
		dataAccessStruct := <-p.DataAccessChannel
		// if it is not leaf it propagated the message to the its children
		if !p.IsLeaf() {
			if err := p.SendToChildren(&dataAccessStruct.DataAccessMessage); err != nil {
				log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String(), ": ", err)
			}
		}
		var dataAccessMap map[string]*training.Examples
		err := json.Unmarshal(dataAccessStruct.DataAccessMessage.TrainData, &dataAccessMap)
		if err != nil {
			log.Fatal("Unmarshalling failed with error: ", err)
		}
		trainData := *dataAccessMap[p.TreeNode().ID.String()]
		train, val = trainData.Split(0.8)
		log.Lvl2("train size: ", len(train))
		log.Lvl2("validation size: ", len(val))
		return train, val
	}

	var test training.Examples
	var err error
	if p.IsRoot() {
		test, err = dataloader.Load(p.ProtocolConf.Dataset, p.ProtocolConf.TestSet)
		log.Lvl2("test size: ", len(test))
		if err != nil {
			panic(err)
		}
	}
	return test, nil
}

//buildGlobalWeights add up ponderated aggregated weights and marshal it
func (p *NeuralNetProtocol) buildGlobalWeights(net *deep.Neural, aggregatedWeights [][][]float64, trainSize float64) []byte {
	fedWeights := federatedWeights(aggregatedWeights, trainSize)
	p.GlobalWeights = fedWeights
	net.ApplyWeights(fedWeights)

	globalWeightsToSend, err := marshalWeights(p.GlobalWeights)
	if err != nil {
		log.Fatal("json marshalling failed with error: ", err)
	}

	return globalWeightsToSend
}

//newIterationAnnouncementPhase Receives a new iteration message and forward it to the children
func (p *NeuralNetProtocol) newIterationAnnouncementPhase() NewIterationMessage {
	// wait for the message from the root to start the protocol
	newIterationMessage := <-p.AnnouncementChannel
	// if p is not leaf it propagated the message to the children
	if !p.IsLeaf() {
		if err := p.SendToChildren(&newIterationMessage.NewIterationMessage); err != nil {
			log.Fatal("Error sending <ForwardPassMessage>: ", p.ServerIdentity().String())
		}
	}
	return newIterationMessage.NewIterationMessage
}

// federatedWeights divide all the parameters's value with the size of the whole training set
func federatedWeights(weights [][][]float64, trainSize float64) [][][]float64 {
	fedWeights := weights
	nu := trainSize
	for i := range fedWeights {
		for j := range fedWeights[i] {
			for k := range fedWeights[i][j] {
				fedWeights[i][j][k] /= nu
			}
		}
	}
	return fedWeights
}

//ponderateParameters ponderate the parameters' value with the size of the local trainin set
func ponderateParameters(weights [][][]float64, subTrainSize int) [][][]float64 {
	ponderatedParameters := weights
	nk := float64(subTrainSize)
	for i := range ponderatedParameters {
		for j := range ponderatedParameters[i] {
			for k := range ponderatedParameters[i][j] {
				ponderatedParameters[i][j][k] *= nk
			}
		}
	}
	return ponderatedParameters
}

//drain set all the values from a pointer to a [][][]float64 to 0
func drain(m *[][][]float64) {
	for i := 0; i < len(*m); i++ {
		for j := 0; j < len((*m)[i]); j++ {
			for k := 0; k < len((*m)[i][j]); k++ {
				(*m)[i][j][k] = 0.
			}
		}
	}
}

//add add two float64 slices
func add(a, b []float64) []float64 {
	sum := make([]float64, len(a))
	for i, v := range a {
		sum[i] = v + b[i]
	}
	return sum
}

//add2d add up two float64 2d slices
func add2d(a, b [][]float64) [][]float64 {
	sum := make([][]float64, len(a))
	for i, v := range a {
		sum[i] = make([]float64, len(v))
	}
	for i, v := range a {
		sum[i] = add(v, b[i])
	}
	return sum
}

//add3d add up two float64 3d slices
func add3d(a, b [][][]float64) [][][]float64 {
	sum := a
	for i := range a {
		sum[i] = add2d(sum[i], b[i])
	}
	return sum
}

func marshalWeights(w [][][]float64) ([]byte, error) {
	byteWeights, err := json.Marshal(w)
	return byteWeights, err
}

func unmarshalWeights(data []byte, w *[][][]float64) error {
	err := json.Unmarshal(data, w)
	return err
}
