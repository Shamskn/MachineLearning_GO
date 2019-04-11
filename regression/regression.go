package regression

import "gonum.org/v1/gonum/mat"

type Regression interface {
	Fit()
	Predict() *mat.VecDense
}
