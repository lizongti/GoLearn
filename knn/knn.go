package knn

import (
	"fmt"

	"github.com/lizongti/golearn/display"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func Run(dataPath string) {
	rawData, err := base.ParseCSVToInstances(dataPath, false)
	if err != nil {
		panic(err)
	}
	display.Section("rawData", rawData)

	// Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	display.Section("trainData", trainData)
	display.Section("testData", testData)

	cls := knn.NewKnnClassifier("euclidean", "linear", 2)
	display.Section("classifier", cls)

	err = cls.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}
	display.Line()

	display.Section("predictions", predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	display.Section("summary", evaluation.GetSummary(confusionMat))
}
