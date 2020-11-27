package dataloader

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"github.com/patrikeh/go-deep/training"
)

//Load load the dataset at path and return a set of examples for go-deep
//library to digest data with neural network model
func Load(dataset string, path string) (training.Examples, error) {

	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}

	var toExample func([]string) training.Example

	if dataset == "mnist" {
		toExample = mnistToExample
	} else if dataset == "bc" {
		toExample = bcToExample
	} else if dataset == "diabete" {
		toExample = diabeteToExample
	}

	r := csv.NewReader(bufio.NewReader(f))

	var examples training.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

//bcToExample CSV parser for breast cancer dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
func bcToExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[9], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(2, res)
	var features []float64

	for i := 0; i < len(in)-1; i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

//mnistToExample CSV parser for mnist data set
func mnistToExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[len(in)-1], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(10, res)
	var features []float64
	for i := 0; i < len(in)-1; i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

//diabeteToExample CSV parser for diabete data set from: https://www.kaggle.com/brandao/diabetes
func diabeteToExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[len(in)-1], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(2, res)
	var features []float64

	for i := 0; i < len(in)-1; i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)] = 1
	return res
}
