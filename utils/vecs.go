package utils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func Ones(n int) *mat.VecDense {
	ones := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		ones.SetVec(i, 1)
	}
	return ones
}

type ExtendedVecDense struct {
	mat.VecDense
}

func (evd *ExtendedVecDense) ScalarAdd(val float64) {
	r, _ := evd.Dims()
	for i := 0; i < r; i++ {
		evd.SetVec(i, val+evd.At(i, 0))
	}
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n\n", fa)
}
