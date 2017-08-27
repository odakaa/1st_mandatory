//Ported by Tengel Ekrem Skar
//A port of Einar Holsbøs logistic regression implementation in python
package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

//This function is identical to that of classifier.go, parallelize it
func classifierConcurrent() float64 {
	batches := 100
	dataRows, dataCols := data.Dims()
	answerRows, answerCols := answers.Dims()
	//subtract 500 elements from the data rows since 500 are for test purposes
	batchSize := (dataRows - 500) / batches
	if debug {
		fmt.Println("Batchsize=", batchSize)
		fmt.Println("Data dims r=", dataRows, " c=", dataCols)
		fmt.Println("Answer dims r=", answerRows, " c=", answerCols)
	}
	//keep first 500 samples for validation
	testX := mat.DenseCopyOf(data.Slice(0, 499, 0, lastcol)) //0...499
	testY := answers.SliceVec(0, 499)
	//x is the training set
	x := mat.DenseCopyOf(data.Slice(500, dataRows, 0, lastcol)) //500...:
	//y is a vecdense(column vector) containing all classifications from 500:...end
	y := answers.SliceVec(500, answerRows)
	//in golang, all variables are initialized to their zero-value.
	coefs := make([]float64, lastcol)

	//==========TRAINING STAGE====================
	for iteration := 0; iteration < 500; iteration++ {
		//each iteration is a single gradient descent step
		for j := 0; j < batches; j++ {
			subsetX := x.Slice(j*batchSize, (j+1)*batchSize, 0, lastcol)
			subsetY := y.SliceVec(j*batchSize, (j+1)*batchSize)

			for i := 0; i < lastcol; i++ {
				coefs[i] = coefs[i] - learningRate*gradient(subsetX, subsetY, coefs, i)
			}
		}
		diff := calcDiff(truth, coefs)
		fmt.Println("Iteration ", iteration+1, " of 500. Diff=", diff)
		//fmt.Println("Iteration ", iteration+1, " of 500.")
	}
	//=========TEST STAGE================//
	fmt.Println("Finished training, moving to test set")
	predictionsV := probGivenX(testX, mat.NewVecDense(len(coefs), coefs))
	predictions := mat.Col(nil, 0, predictionsV)
	testAnswers := mat.Col(nil, 0, testY)
	var correct = 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] > 0.5 {
			predictions[i] = 1
		} else {
			predictions[i] = 0
		}
		//because we want high accuracy
		prayforzero := predictions[i] - testAnswers[i]
		if math.Abs(prayforzero) < 0.01 {
			correct++
		}
	}
	accuracy := (float64(correct) / float64(len(predictions)))
	fmt.Println("Accuracy:", accuracy, "%")
	//fmt.Println(coefs)
	return accuracy
}
