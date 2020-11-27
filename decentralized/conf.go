package decentralized

import (
	"encoding/json"
	"log"
	"os"
)

type conf struct {
	Dataset                        string
	StatWritingFolderDecentralized string
	ExperimentName                 string
	TrainSet                       string
	TestSet                        string
	ExpRep                         int
	InputSize                      int
	Layout                         []int
	Activation                     string
	ActivationType                 string
	MaxIteration                   int
	NodeNumber                     int
	BatchSize                      int
	LearningRate                   float64
	Momentum                       float64
	Decay                          float64
	Nesterov                       bool
	Epochs                         int
	FailureRate                    float64
	FullDeactivationRate           float64
	DeactivatedServer              string
	VerboseTrain                   int
	NClass                         int
	TrainSize                      float64
}

//GetConf read a config file and build a conf struct
func (c *conf) GetConf(filename string) {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal("Config not initialized: ", err)
	}

	decoder := json.NewDecoder(file)
	err = decoder.Decode(&c)
	if err != nil {
		log.Fatal("config file not decoded: ", err)
	}
}
