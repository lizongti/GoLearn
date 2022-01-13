package knn

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func Run(dataPath string) {
	rawData, err := base.ParseCSVToInstances(dataPath, false)
	if err != nil {
		panic(err)
	}
	display("rawData", rawData)

	cls := knn.NewKnnClassifier("euclidean", "linear", 2)
	display("classifier", cls)

	// Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	display("trainData", trainData)
	display("testData", testData)

	err = cls.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}
	line()

	display("predictions", predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	display("summary", evaluation.GetSummary(confusionMat))
}

func line() {
	fmt.Println("")
}

var count int = 0

func display(s string, v interface{}) {
	count++
	fmt.Printf("\n----------------------------[%d.%s]----------------------------\n%v\n", count, s, v)
}
