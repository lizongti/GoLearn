package art

import (
	"fmt"

	"github.com/lizongti/golearn/display"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
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

	decTree := trees.NewDecisionTreeClassifier("entropy", -1, []int64{0, 1})
	display.Section("classifier", decTree)

	err = decTree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	display.Section("decTree", decTree)

	// Access Predictions
	predictions := decTree.Predict(testData)
	display.Section("predictioins", predictions)

	// cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	// if err != nil {
	// 	panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	// }
	// fmt.Println(evaluation.GetSummary(cf))
}
