package stats

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

// Writer prints training progress
type Writer struct {
	base   string
	file   *os.File
	writer *csv.Writer
}

// NewStatsWriter creates a StatsPrinter
func NewStatsWriter(base string) *Writer {
	return &Writer{base, new(os.File), new(csv.Writer)}
}

// Init initializes printer
func (w *Writer) Init(root bool, id string) {
	if _, err := os.Stat(w.base); os.IsNotExist(err) {
		os.Mkdir(w.base, os.ModePerm)
	}
	if root {
		if _, err := os.Stat("/master.csv"); os.IsNotExist(err) {
			file, err := os.Create(w.base + "/master.csv")
			if err != nil {
				log.Fatal(err)
			}
			w.file = file
			w.writer = csv.NewWriter(w.file)
		}
	} else {
		if _, err := os.Stat("/" + id + ".csv"); os.IsNotExist(err) {
			file, err := os.Create(w.base + "/" + id + ".csv")
			if err != nil {
				log.Fatal(err)
			}
			w.file = file
			w.writer = csv.NewWriter(w.file)
		}
	}

}

func (w *Writer) Write(value []float64) {
	values := make([]string, len(value))
	for i, e := range value {
		values[i] = fmt.Sprintf("%f", e)
	}
	w.writer.Write(values)
	w.writer.Flush()
}

// WriteProgress prints the current state of training
func (w *Writer) WriteProgress(net *deep.Neural, validation training.Examples, NumberOfClasses int) {
	w.writer.Write([]string{formatAccuracy(net, validation, NumberOfClasses)})
	w.writer.Flush()
}

func formatAccuracy(net *deep.Neural, validation training.Examples, NumberOfClasses int) string {
	return fmt.Sprintf("%.2f\t", Accuracy(net, validation))
}

// Loss return the loss of a dataset
func Loss(n *deep.Neural, validation training.Examples) float64 {

	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return deep.GetLoss(n.Config.Loss).F(predictions, responses)
}

// Accuracy metric for classification
func Accuracy(n *deep.Neural, validation training.Examples) float64 {
	correct := 0
	for i, e := range validation {
		est := n.Predict(validation[i].Input)
		if deep.ArgMax(e.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation))
}

// PredictionVsActual returns both the vectors of prediction and actual values
func PredictionVsActual(n *deep.Neural, validation training.Examples) (predicted, actual []int64) {
	predicted = make([]int64, len(validation))
	actual = make([]int64, len(validation))
	for i, e := range validation {
		est := n.Predict((e.Input))
		predicted[i] = int64(deep.ArgMax(est))
		actual[i] = int64(deep.ArgMax(e.Response))
	}
	return
}
