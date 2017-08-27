//Authored by Tengel Ekrem Skar
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

var debug = false
var bigData = false
var data *mat.Dense
var answers *mat.VecDense
var lastcol int
var truth []float64
var learningRate = 0.005

//in go, init is always run prior to execution of main
func init() {
	fmt.Println("Opening data set, big data set to:", bigData)
	if bigData {
		lastcol = 29
		data, answers = preProcessFile("./creditfraud.csv")
		truth = []float64{0.144800089, 4.537449930, 5.246915267, 7.115031732, 2.518657741,
			1.545085627, -2.830038407, -2.410473992, 34.786359072,
			-9.919927002, 23.812173327, -61.670817783, 40.189039662,
			33.862622820, -16.615272064, -0.618733376, 15.309273546,
			5.631484914, -4.066823502, -0.888067511, -2.755034830,
			-3.322881536, 0.162538846, -0.916925947, -0.288467553,
			-0.456699696, -0.304502125, 0.008763563, 0.010576039}
	} else {
		lastcol = 57
		data, answers = preProcessFile("./spam.csv")
		truth = []float64{-1.492773e-01, -2.405872e-01, -1.598794e-01, 1.792132e-02,
			1.521252e+00, 3.116990e-02, 5.117602e-01, 1.328038e-01,
			9.483219e-02, 4.187917e-01, 7.968122e-01, -7.066990e-01,
			-5.672777e-01, -4.229411e-02, 1.217488e+01, -6.331285e-02,
			1.828883e+00, 2.489344e-02, 1.059606e-01, 3.472549e+00,
			2.804220e-01, 5.623414e-01, 1.025395e+00, 5.151430e-01,
			-8.775583e+00, -1.929012e-01, -1.384527e+03, 1.620060e+00,
			4.331880e-01, -5.719369e-01, -9.152286e+01, -1.425805e+05,
			-2.991609e-01, 1.430954e+05, -8.514605e+00, 2.089551e-01,
			-1.557250e+00, 9.149123e+01, 4.663478e-01, -3.890767e-01,
			-1.601655e+02, -7.735470e+01, -1.787178e+00, -2.385616e+01,
			-6.600457e-01, -9.924051e-01, -1.279255e+00, -1.400962e+00,
			-5.165253e-01, -1.250286e-02, -4.783125e-01, 1.081903e+00,
			1.688540e+00, -1.991582e+00, 1.464189e+01, -3.194082e+00,
			4.274010e+00}
	}
	fmt.Println("Done")
}

//convert all strings to floats, take last column and add to its own array of integers
func convertStringArrToMatrix(string2dArr [][]string) (*mat.Dense, *mat.VecDense) {
	var err error
	var i, j int
	var vectors *mat.Dense
	var classes *mat.VecDense
	var classesF []float64
	var floatArr []float64
	//temp storage for each vector before it is converted to Dense

	classesF = make([]float64, len(string2dArr))
	floatArr = make([]float64, lastcol*len(string2dArr))

	if debug {
		fmt.Println("data set rows: ", len(string2dArr))
		fmt.Println("data set columns: ", len(string2dArr[0]))
		fmt.Printf("Converting and storing vectors...")
	}

	for i = 0; i < len(string2dArr); i++ {
		//convert string array to float array
		for j = 0; j < lastcol; j++ {
			floatArr[i*lastcol+j], err = strconv.ParseFloat(string2dArr[i][j], 64)
			if err != nil {
				log.Fatal(err)
			}
		}
		classesF[i], err = strconv.ParseFloat(string2dArr[i][lastcol], 64)
		if err != nil {
			log.Fatal(err)
		}
	}
	vectors = mat.NewDense(len(string2dArr), lastcol, floatArr)
	classes = mat.NewVecDense(len(classesF), classesF)

	return vectors, classes
}

//Parse the file, convert all entries to FP, return a 2D array with all random vectors
//and a vector of classifications ([]float6)
func parseFile(filename string) [][]string {
	var string2DArr [][]string

	r, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	csvparser := csv.NewReader(r)
	string2DArr, err = csvparser.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	return string2DArr
}
func preProcessFile(filename string) (*mat.Dense, *mat.VecDense) {
	var string2DArr [][]string
	string2DArr = parseFile(filename)
	vectors, classes := convertStringArrToMatrix(string2DArr)
	return vectors, classes
}

func main() {
	startConcurrent := time.Now()
	concurrentAccuracy := classifierConcurrent()
	elapsedConcurrent := time.Since(startConcurrent)
	startSingle := time.Now()
	singleAccuracy := classifierSingle()
	elapsedSingle := time.Since(startSingle)
	fmt.Println("==Concurrent implementation==")
	fmt.Println("Time:", elapsedConcurrent)
	fmt.Println("Accuracy:", concurrentAccuracy)
	fmt.Println("==Singlethread implementation==")
	fmt.Println("Time:", elapsedSingle)
	fmt.Println("Accuracy:", singleAccuracy)
	fmt.Println("==Speedup: ", (elapsedSingle.Seconds() / elapsedConcurrent.Seconds()), " ==")
}
