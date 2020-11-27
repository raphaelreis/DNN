package centralized

import (
	"encoding/json"
	"log"
	"os"
)

type conf struct {
	Dataset                      string
	StatWritingFolderCentralized string
	ExperimentName               string
	TrainSet                     string
	TestSet                      string
	InputSize                    int
	ExpRep                       int
	Layout                       []int
	LearningRate                 float64
	Momentum                     float64
	Decay                        float64
	Nesterov                     bool
	Epochs                       int
	NClass                       int
	BatchSize                    int
	VerboseTrain                 int
	Activation                   string
	ActivationType               string
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
